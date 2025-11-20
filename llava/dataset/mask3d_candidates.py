#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update Mask3D prediction instances with:
1. 3D bounding boxes (bbox8 - 8 corner points)
2. 2D projections onto multiple camera views (bbox_2d, ranging_ratio, vis_ratio)

ranging_ratio: ratio of instance points within camera view frustum (no occlusion test)
vis_ratio: ratio of instance points actually visible without occlusion (using depth maps)
"""

import os
import re
import glob
import shutil
import argparse
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# Default config file path (relative to this script)
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "base.yaml")


# ============================================================
# 1. 3D BBox Computation (from evaluate_parsed_description.py)
# ============================================================

def compute_bbox8(points):
    """
    Compute axis-aligned 3D bounding box (8 corner points) from point cloud

    Args:
        points: (N, 3) or (N, >3) array of points

    Returns:
        corners: (8, 3) array of corner coordinates
    """
    if points.shape[1] > 3:
        points = points[:, :3]
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)

    # Generate 8 corners of axis-aligned cube
    corners = np.array([
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
    ])
    return corners


# ============================================================
# 2. Scene Metadata Reading (from visual_bbox_2Dviews.py)
# ============================================================

_FLOAT_RE = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

def read_scene_txt_color_only(scene_dir):
    """
    Read scene .txt file to extract:
      - axisAlignment (4x4 matrix)
      - color camera intrinsics (fx, fy, mx, my, width, height)

    Returns:
        M_align: (4, 4) alignment matrix
        intr: dict of intrinsic parameters
    """
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    txt_path = os.path.join(scene_dir, f"{scene_id}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    norm = (content.replace('\r', ' ')
                   .replace('\u00A0',' ')
                   .replace('\u200b',' '))
    norm = re.sub(r'[ \t]+', ' ', norm)

    def grab_mat4(name):
        m = re.search(rf'{name}\s*=\s*((?:{_FLOAT_RE}\s+){{15}}{_FLOAT_RE})', norm, re.I)
        if not m:
            raise ValueError(f"{name} not found or malformed in {txt_path}")
        vals = [float(x) for x in re.findall(_FLOAT_RE, m.group(1))]
        return np.array(vals, dtype=np.float64).reshape(4,4)

    def grab_scalar(name, cast=float):
        m = re.search(rf'\b{name}\b\s*=\s*({_FLOAT_RE})', norm, re.I)
        if not m:
            raise ValueError(f"{name} not found in {txt_path}")
        return cast(m.group(1))

    M_align = grab_mat4("axisAlignment")

    intr = {
        "fx_color": grab_scalar("fx_color"),
        "fy_color": grab_scalar("fy_color"),
        "mx_color": grab_scalar("mx_color"),
        "my_color": grab_scalar("my_color"),
        "colorWidth":  int(grab_scalar("colorWidth",  float)),
        "colorHeight": int(grab_scalar("colorHeight", float)),
    }
    return M_align, intr


def make_K_color_scaled(intr, W_img, H_img):
    """
    Create scaled camera intrinsic matrix K for given image dimensions

    Args:
        intr: dict with fx_color, fy_color, mx_color, my_color, colorWidth, colorHeight
        W_img: actual image width
        H_img: actual image height

    Returns:
        K: (3, 3) camera intrinsic matrix
    """
    sx, sy = W_img / intr["colorWidth"], H_img / intr["colorHeight"]
    return np.array([
        [intr["fx_color"] * sx, 0,                    intr["mx_color"] * sx],
        [0,                    intr["fy_color"] * sy, intr["my_color"] * sy],
        [0,                    0,                    1.0]
    ], dtype=np.float64)


# ============================================================
# 3. Depth Map Loading
# ============================================================

def load_depth_map(depth_path, scale=1000.0):
    """
    Load depth map from PNG file and convert to meters

    Args:
        depth_path: path to depth PNG file
        scale: scale factor to convert to meters (default 1000.0 for mm to m)

    Returns:
        depth: (H, W) depth map in meters, or None if file doesn't exist
    """
    if not os.path.exists(depth_path):
        return None

    try:
        depth_img = np.array(Image.open(depth_path), dtype=np.float32)
        # Convert to meters
        depth = depth_img / scale
        # Set invalid depths to 0
        depth[depth < 0] = 0
        return depth
    except Exception as e:
        print(f"[WARNING] Failed to load depth map {depth_path}: {e}")
        return None


# ============================================================
# 4. Camera Pose & Projection (from visual_bbox_2Dviews.py)
# ============================================================

def load_pose_txt(pose_txt):
    """
    Load camera-to-world pose from .txt file

    Returns:
        T_c2w: (4, 4) camera-to-world transformation matrix
    """
    vals = []
    with open(pose_txt, "r") as f:
        for line in f:
            vals += [float(x) for x in line.strip().replace(",", " ").split()]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Invalid pose: {pose_txt}")
    T = arr.reshape(4, 4)

    # Validate that the pose contains finite values
    if not np.isfinite(T).all():
        raise ValueError(f"Pose contains NaN/inf: {pose_txt}")

    return T


def project_points_and_stats_color(K, T_w2c, pts_world, W, H, depth_map=None,
                                   min_z=0.05, max_z=80.0, depth_tolerance=0.05):
    """
    Project 3D points to 2D and compute visibility statistics with occlusion handling

    Args:
        K: (3, 3) camera intrinsic matrix
        T_w2c: (4, 4) world-to-camera transformation
        pts_world: (N, 3) or (N, >3) 3D instance points in world coordinates (aligned)
        W, H: image dimensions
        depth_map: (H, W) depth image in meters, for occlusion testing
        min_z, max_z: depth filtering range
        depth_tolerance: tolerance for depth comparison (in meters)

    Returns:
        mask: (H, W) binary mask of visible pixels
        rect: (x1, y1, x2, y2) 2D bounding box
        ranging_ratio: points within view frustum / total points
        vis_ratio: points visible without occlusion / total points
        ranging_count: number of points within view frustum
        vis_count: number of visible points (not occluded)
        total: total number of points
    """
    # Ensure we only use xyz coordinates
    if pts_world.shape[1] > 3:
        pts_world = pts_world[:, :3]

    total = pts_world.shape[0]

    # Project instance points
    homog = np.hstack([pts_world, np.ones((total, 1), dtype=np.float64)])
    cam = (T_w2c @ homog.T).T
    xyz = cam[:, :3]
    z = xyz[:, 2]
    valid = (z > min_z) & (z < max_z)
    if not np.any(valid):
        return (None, None, 0.0, 0.0, 0, 0, total)

    xyzv = xyz[valid]
    zv = z[valid]
    uvh = (K @ xyzv.T).T
    uv = uvh[:, :2] / zv[:, None]

    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ranging_count = int(np.count_nonzero(keep))
    if ranging_count == 0:
        return (None, None, 0.0, 0.0, 0, 0, total)

    u_in, v_in = u[keep], v[keep]
    z_in = zv[keep]

    # Test occlusion for instance points using provided depth map
    if depth_map is not None:
        # Check if instance points are occluded
        visible_mask = np.zeros(len(u_in), dtype=bool)
        for i in range(len(u_in)):
            uu, vv, zz = u_in[i], v_in[i], z_in[i]
            scene_depth = depth_map[vv, uu]
            # Point is visible if its depth is close to the scene depth at that pixel
            # Also check that scene_depth is valid (> 0)
            if scene_depth > 0 and zz <= scene_depth + depth_tolerance:
                visible_mask[i] = True

        vis_count = int(np.count_nonzero(visible_mask))
        u_vis = u_in[visible_mask]
        v_vis = v_in[visible_mask]
    else:
        # No occlusion testing, all in-range points are considered visible
        vis_count = ranging_count
        u_vis = u_in
        v_vis = v_in

    if vis_count == 0:
        return (None, None, 0.0, 0.0, ranging_count, 0, total)

    # Create mask from visible points
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[v_vis, u_vis] = 255

    ys, xs = np.where(mask > 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    rect = (float(x1), float(y1), float(x2), float(y2))

    ranging_ratio = float(ranging_count) / float(total)
    vis_ratio = float(vis_count) / float(total)

    return (mask, rect, ranging_ratio, vis_ratio, ranging_count, vis_count, total)


# ============================================================
# 5. Main Update Function
# ============================================================

def update_scene_with_bbox_and_views(
    scene_id,
    PRED_FILE,
    scans_root,
    color_root,
    SAVE_ROOT=None,
    min_pixels=120,
    min_ratio=0.0005,
    max_points=200000
):
    """
    Update Mask3D prediction with:
    1. 3D bbox (8 corners)
    2. 2D projections on multiple views (bbox_2d, ranging_ratio, vis_ratio)

    This function uses pre-computed depth maps from {scene_id}/depth_inpainted/*.png
    for occlusion testing.

    Two visibility ratios are computed:
    - ranging_ratio: ratio of instance points within the camera view frustum
    - vis_ratio: ratio of instance points actually visible (not occluded by depth map)

    Args:
        scene_id: scene identifier
        PRED_FILE: directory containing {scene_id}.npz predictions (aligned coords)
        scans_root: directory containing scene .txt files with metadata
        color_root: directory containing {scene_id}/color/*.png, pose/*.txt, and depth_inpainted/*.png
        SAVE_ROOT: output directory (default: PRED_FILE)
        min_pixels: minimum visible pixels to consider a view valid
        min_ratio: minimum visibility ratio to consider a view valid
        max_points: downsample instance points to this number

    Returns:
        None (moves original to backup/ and saves to {scene_id}.npz)
    """
    # Load prediction
    pred_path = os.path.join(PRED_FILE, f"{scene_id}.npz")
    if not os.path.exists(pred_path):
        print(f"[WARNING] Prediction not found: {pred_path}")
        return

    instance_pred = np.load(pred_path, allow_pickle=True)

    # Check if already updated (has bbox8 and views info)
    if 'bbox8' in instance_pred and 'views_per_instance' in instance_pred:
        print(f"[SKIP] {scene_id}: already updated with bbox8 and views info")
        return

    ins_pcds = instance_pred["ins_pcds"]
    ins_labels = instance_pred["ins_labels"]
    ins_scores = instance_pred["ins_scores"]

    # 1. Compute 3D bbox8 for each instance
    bbox8_list = []
    for pts in ins_pcds:
        if len(pts) == 0:
            bbox8 = np.zeros((8, 3))
        else:
            bbox8 = compute_bbox8(pts)
        bbox8_list.append(bbox8)
    bbox8_array = np.stack(bbox8_list, axis=0)  # (K, 8, 3)

    # 2. Read scene metadata
    scene_dir = os.path.join(scans_root, scene_id)
    try:
        M_align, intr = read_scene_txt_color_only(scene_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[WARNING] Failed to read scene metadata for {scene_id}: {e}")
        # Save with 3D bbox only
        if SAVE_ROOT is None:
            SAVE_ROOT = PRED_FILE
        os.makedirs(SAVE_ROOT, exist_ok=True)
        # Backup original file if exists
        backup_dir = os.path.join(SAVE_ROOT, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{scene_id}.npz")
        if os.path.exists(pred_path) and not os.path.exists(backup_path):
            shutil.move(pred_path, backup_path)
        save_path = os.path.join(SAVE_ROOT, f"{scene_id}.npz")
        np.savez_compressed(
            save_path,
            ins_pcds=ins_pcds,
            ins_labels=ins_labels,
            ins_scores=ins_scores,
            bbox8=bbox8_array
        )
        return

    # 3. List all available views
    color_dir = os.path.join(color_root, scene_id, "color")
    if not os.path.exists(color_dir):
        print(f"[WARNING] Color directory not found: {color_dir}")
        # Save with 3D bbox only
        if SAVE_ROOT is None:
            SAVE_ROOT = PRED_FILE
        os.makedirs(SAVE_ROOT, exist_ok=True)
        # Backup original file if exists
        backup_dir = os.path.join(SAVE_ROOT, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{scene_id}.npz")
        if os.path.exists(pred_path) and not os.path.exists(backup_path):
            shutil.move(pred_path, backup_path)
        save_path = os.path.join(SAVE_ROOT, f"{scene_id}.npz")
        np.savez_compressed(
            save_path,
            ins_pcds=ins_pcds,
            ins_labels=ins_labels,
            ins_scores=ins_scores,
            bbox8=bbox8_array
        )
        return

    view_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    views = [os.path.splitext(os.path.basename(p))[0] for p in view_files]

    if not views:
        print(f"[WARNING] No color frames found for {scene_id}")
        # Save with 3D bbox only
        if SAVE_ROOT is None:
            SAVE_ROOT = PRED_FILE
        os.makedirs(SAVE_ROOT, exist_ok=True)
        # Backup original file if exists
        backup_dir = os.path.join(SAVE_ROOT, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{scene_id}.npz")
        if os.path.exists(pred_path) and not os.path.exists(backup_path):
            shutil.move(pred_path, backup_path)
        save_path = os.path.join(SAVE_ROOT, f"{scene_id}.npz")
        np.savez_compressed(
            save_path,
            ins_pcds=ins_pcds,
            ins_labels=ins_labels,
            ins_scores=ins_scores,
            bbox8=bbox8_array
        )
        return

    # 4. Project each instance onto each view
    # Structure: views_per_instance[i] = list of view IDs where instance i is visible
    # Structure: bbox2d_per_instance[i] = {view_id: [x1, y1, x2, y2], ...}
    # Structure: ranging_ratio_per_instance[i] = {view_id: ranging_ratio, ...}
    # Structure: vis_ratio_per_instance[i] = {view_id: vis_ratio, ...}
    # Structure: instances_per_view = {view_id: [inst_idx1, inst_idx2, ...], ...}

    views_per_instance = []
    bbox2d_per_instance = []
    ranging_ratio_per_instance = []
    vis_ratio_per_instance = []
    instances_per_view = {v: [] for v in views}  # Initialize with all views

    for inst_idx, pts3d in enumerate(ins_pcds):
        if len(pts3d) == 0:
            views_per_instance.append([])
            bbox2d_per_instance.append({})
            ranging_ratio_per_instance.append({})
            vis_ratio_per_instance.append({})
            continue

        # Downsample if needed
        if pts3d.shape[0] > max_points:
            idx = np.random.choice(pts3d.shape[0], max_points, replace=False)
            pts3d = pts3d[idx]

        pos_views = []
        bbox_per_view = {}
        ranging_ratio_per_view = {}
        vis_ratio_per_view = {}

        for v in views:
            img_path = os.path.join(color_dir, f"{v}.png")
            pose_txt = os.path.join(color_root, scene_id, "pose", f"{v}.txt")
            depth_path = os.path.join(color_root, scene_id, "depth_inpainted", f"{v}.png")

            if not (os.path.exists(img_path) and os.path.exists(pose_txt)):
                continue

            try:
                with Image.open(img_path) as im0:
                    W, H = im0.size
                T_c2w = load_pose_txt(pose_txt)
            except Exception:
                continue

            # Load depth map for occlusion testing
            depth_map = load_depth_map(depth_path)

            # Color camera: scale K, T_w2c = inv(M_align @ T_c2w)
            K = make_K_color_scaled(intr, W, H)
            T_c2w_aligned = M_align @ T_c2w
            T_w2c = np.linalg.inv(T_c2w_aligned)

            # Project points
            result = project_points_and_stats_color(
                K, T_w2c, pts3d, W, H, depth_map=depth_map,
                min_z=0.05, max_z=80.0
            )

            mask, rect, ranging_ratio, vis_ratio, ranging_cnt, vis_cnt, total_cnt = result

            if mask is None or rect is None:
                continue
            # Use vis_cnt (actual visible points) for filtering
            if vis_cnt < min_pixels or vis_ratio < min_ratio:
                continue

            pos_views.append(v)
            bbox_per_view[v] = [float(f"{rect[0]:.2f}"), float(f"{rect[1]:.2f}"),
                               float(f"{rect[2]:.2f}"), float(f"{rect[3]:.2f}")]
            ranging_ratio_per_view[v] = float(f"{ranging_ratio:.6f}")
            vis_ratio_per_view[v] = float(f"{vis_ratio:.6f}")
            instances_per_view[v].append(inst_idx)  # Record instance in this view

        views_per_instance.append(sorted(pos_views, key=lambda x: int(x) if x.isdigit() else x))
        bbox2d_per_instance.append(dict(sorted(bbox_per_view.items(),
                                              key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])))
        ranging_ratio_per_instance.append(dict(sorted(ranging_ratio_per_view.items(),
                                                     key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])))
        vis_ratio_per_instance.append(dict(sorted(vis_ratio_per_view.items(),
                                                  key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])))

    # 5. Sort instances_per_view by view ID and convert to sorted dict
    instances_per_view = dict(sorted(instances_per_view.items(),
                                    key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))

    # 6. Save updated file
    if SAVE_ROOT is None:
        SAVE_ROOT = PRED_FILE
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Backup original file if exists
    backup_dir = os.path.join(SAVE_ROOT, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"{scene_id}.npz")
    if os.path.exists(pred_path) and not os.path.exists(backup_path):
        shutil.move(pred_path, backup_path)

    save_path = os.path.join(SAVE_ROOT, f"{scene_id}.npz")
    np.savez_compressed(
        save_path,
        ins_pcds=ins_pcds,
        ins_labels=ins_labels,
        ins_scores=ins_scores,
        bbox8=bbox8_array,
        views_per_instance=np.array(views_per_instance, dtype=object),
        bbox2d_per_instance=np.array(bbox2d_per_instance, dtype=object),
        ranging_ratio_per_instance=np.array(ranging_ratio_per_instance, dtype=object),
        vis_ratio_per_instance=np.array(vis_ratio_per_instance, dtype=object),
        instances_per_view=np.array(instances_per_view, dtype=object)
    )

    print(f"[✓] {scene_id}: {len(ins_pcds)} instances with bbox8 + 2D views (ranging_ratio + vis_ratio) -> {save_path}")


# ============================================================
# 6. Parallel Processing Wrapper
# ============================================================

def process_scene_wrapper(scene_id, args_dict):
    """
    Wrapper function for parallel processing of scenes

    Args:
        scene_id: scene identifier
        args_dict: dictionary containing all arguments
    """
    try:
        update_scene_with_bbox_and_views(
            scene_id=scene_id,
            PRED_FILE=args_dict['pred_file'],
            scans_root=args_dict['scans_root'],
            color_root=args_dict['color_root'],
            SAVE_ROOT=args_dict['save_root'],
            min_pixels=args_dict['min_pixels'],
            min_ratio=args_dict['min_ratio'],
            max_points=args_dict['max_points']
        )
        return scene_id, True
    except Exception as e:
        print(f"[ERROR] Failed to process {scene_id}: {e}")
        return scene_id, False


# ============================================================
# 7. Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Update Mask3D predictions with 3D bbox and 2D view projections"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE,
                        help="Path to config YAML file (default: scannetv2.yaml)")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Dataset split to process")
    parser.add_argument("--save_root", default=None,
                        help="Output directory (default: same as mask3d_pred_path)")
    parser.add_argument("--min_pixels", type=int, default=120,
                        help="Minimum visible pixels per view")
    parser.add_argument("--min_ratio", type=float, default=0.1,
                        help="Minimum visibility ratio per view")
    parser.add_argument("--max_points", type=int, default=200000,
                        help="Max points per instance")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of parallel workers (default: 1, sequential)")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract paths from config
    pred_file = cfg["mask3d_pred_path"]
    scans_root = cfg["raw_scene_path"]
    color_root = cfg["color_root"]
    metadata_root = cfg["scene_metadata"]["metadata_root_path"]
    scene_list_file = os.path.join(metadata_root, cfg["scene_metadata"][f"{args.split}_scene_ids"])
    save_root = args.save_root if args.save_root else pred_file

    print(f"Loaded config from: {args.config}")
    print(f"  pred_file: {pred_file}")
    print(f"  scans_root: {scans_root}")
    print(f"  color_root: {color_root}")
    print(f"  scene_list: {scene_list_file}")
    print(f"  save_root: {save_root}")

    # Read scene list
    with open(scene_list_file, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]

    # Prepare arguments dictionary for parallel processing
    args_dict = {
        'pred_file': pred_file,
        'scans_root': scans_root,
        'color_root': color_root,
        'save_root': save_root,
        'min_pixels': args.min_pixels,
        'min_ratio': args.min_ratio,
        'max_points': args.max_points
    }

    print(f"Processing {len(scene_ids)} scenes with {args.num_workers} worker(s)...")

    if args.num_workers > 1:
        # Parallel processing
        print(f"[INFO] Using multiprocessing with {args.num_workers} workers")
        worker_func = partial(process_scene_wrapper, args_dict=args_dict)

        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_func, scene_ids),
                total=len(scene_ids),
                desc="Updating scenes"
            ))

        # Report results
        success_count = sum(1 for _, success in results if success)
        failed_count = len(results) - success_count
        print(f"\n[OK] Processed {len(scene_ids)} scenes: {success_count} succeeded, {failed_count} failed")
        if failed_count > 0:
            failed_scenes = [scene_id for scene_id, success in results if not success]
            print(f"[WARNING] Failed scenes: {failed_scenes}")
    else:
        # Sequential processing
        print(f"[INFO] Using sequential processing")
        for scene_id in tqdm(scene_ids, desc="Updating scenes"):
            update_scene_with_bbox_and_views(
                scene_id=scene_id,
                PRED_FILE=pred_file,
                scans_root=scans_root,
                color_root=color_root,
                SAVE_ROOT=save_root,
                min_pixels=args.min_pixels,
                min_ratio=args.min_ratio,
                max_points=args.max_points
            )

        print(f"\n[OK] All scenes processed!")


if __name__ == "__main__":
    main()
