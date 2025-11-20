#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process ScanNet Ground Truth instances into the same format as Mask3D predictions:
- ins_pcds: Instance point clouds (WITH AXIS ALIGNMENT APPLIED!)
- ins_labels: Instance labels
- ins_scores: Instance confidence scores (all 1.0 for GT)
- bbox8: 3D bounding boxes (8 corners)
- views_per_instance: List of view IDs where each instance is visible
- bbox2d_per_instance: 2D bounding boxes per view
- ranging_ratio_per_instance: Frustum visibility ratios
- vis_ratio_per_instance: Occlusion-aware visibility ratios
- instances_per_view: Which instances are visible in each view
"""

import os
import re
import json
import glob
import argparse
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from plyfile import PlyData

# Default config file path (relative to this script)
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "base.yaml")


# ============================================================
# 1. 3D BBox Computation
# ============================================================

def compute_bbox8(points):
    """
    Compute axis-aligned 3D bounding box (8 corner points) from point cloud
    """
    if points.shape[1] > 3:
        points = points[:, :3]
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)

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
# 2. Scene Metadata Reading
# ============================================================

_FLOAT_RE = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def read_axis_align_matrix(txt_path):
    """Read axis alignment matrix from scene .txt file"""
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    norm = (content.replace('\r', ' ')
                   .replace('\u00A0', ' ')
                   .replace('\u200b', ' '))
    norm = re.sub(r'[ \t]+', ' ', norm)

    m = re.search(rf'axisAlignment\s*=\s*((?:{_FLOAT_RE}\s+){{15}}{_FLOAT_RE})', norm, re.I)
    if not m:
        return None

    vals = [float(x) for x in re.findall(_FLOAT_RE, m.group(1))]
    if len(vals) != 16:
        return None

    return np.array(vals, dtype=np.float64).reshape(4, 4)


def read_scene_txt_color_only(scene_dir):
    """Read scene .txt file to extract alignment and camera intrinsics"""
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    txt_path = os.path.join(scene_dir, f"{scene_id}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    norm = (content.replace('\r', ' ')
                   .replace('\u00A0', ' ')
                   .replace('\u200b', ' '))
    norm = re.sub(r'[ \t]+', ' ', norm)

    def grab_mat4(name):
        m = re.search(rf'{name}\s*=\s*((?:{_FLOAT_RE}\s+){{15}}{_FLOAT_RE})', norm, re.I)
        if not m:
            raise ValueError(f"{name} not found or malformed in {txt_path}")
        vals = [float(x) for x in re.findall(_FLOAT_RE, m.group(1))]
        return np.array(vals, dtype=np.float64).reshape(4, 4)

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
        "colorWidth": int(grab_scalar("colorWidth", float)),
        "colorHeight": int(grab_scalar("colorHeight", float)),
    }
    return M_align, intr


def make_K_color_scaled(intr, W_img, H_img):
    """Create scaled camera intrinsic matrix K"""
    sx, sy = W_img / intr["colorWidth"], H_img / intr["colorHeight"]
    return np.array([
        [intr["fx_color"] * sx, 0, intr["mx_color"] * sx],
        [0, intr["fy_color"] * sy, intr["my_color"] * sy],
        [0, 0, 1.0]
    ], dtype=np.float64)


# ============================================================
# 3. Depth Map Loading
# ============================================================

def load_depth_map(depth_path, scale=1000.0):
    """Load depth map from PNG file and convert to meters"""
    if not os.path.exists(depth_path):
        return None

    try:
        depth_img = np.array(Image.open(depth_path), dtype=np.float32)
        depth = depth_img / scale
        depth[depth < 0] = 0
        return depth
    except Exception as e:
        print(f"[WARNING] Failed to load depth map {depth_path}: {e}")
        return None


# ============================================================
# 4. Camera Pose & Projection
# ============================================================

def load_pose_txt(pose_txt):
    """Load camera-to-world pose from .txt file"""
    vals = []
    with open(pose_txt, "r") as f:
        for line in f:
            vals += [float(x) for x in line.strip().replace(",", " ").split()]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Invalid pose: {pose_txt}")
    T = arr.reshape(4, 4)

    if not np.isfinite(T).all():
        raise ValueError(f"Pose contains NaN/inf: {pose_txt}")

    return T


def project_points_and_stats_color(K, T_w2c, pts_world, W, H, depth_map=None,
                                   min_z=0.05, max_z=80.0, depth_tolerance=0.05):
    """Project 3D points to 2D and compute visibility statistics"""
    if pts_world.shape[1] > 3:
        pts_world = pts_world[:, :3]

    total = pts_world.shape[0]

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

    if depth_map is not None:
        visible_mask = np.zeros(len(u_in), dtype=bool)
        for i in range(len(u_in)):
            uu, vv, zz = u_in[i], v_in[i], z_in[i]
            scene_depth = depth_map[vv, uu]
            if scene_depth > 0 and zz <= scene_depth + depth_tolerance:
                visible_mask[i] = True

        vis_count = int(np.count_nonzero(visible_mask))
        u_vis = u_in[visible_mask]
        v_vis = v_in[visible_mask]
    else:
        vis_count = ranging_count
        u_vis = u_in
        v_vis = v_in

    if vis_count == 0:
        return (None, None, 0.0, 0.0, ranging_count, 0, total)

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
# 5. ScanNet GT Loading Functions
# ============================================================

def load_scannet_gt_instances(scene_dir):
    """
    Load ground truth instances from ScanNet format

    Returns:
        ins_pcds: list of (N, 6) arrays [x, y, z, r, g, b] in aligned coordinates
        ins_labels: list of label strings
        ins_scores: list of confidence scores (all 1.0 for GT)
    """
    scene_id = os.path.basename(scene_dir.rstrip("/"))

    # Read axis alignment matrix
    txt_path = os.path.join(scene_dir, f"{scene_id}.txt")
    M_align = read_axis_align_matrix(txt_path)
    if M_align is None:
        print(f"[WARNING] {scene_id}: No axis alignment matrix found, using identity")
        M_align = np.eye(4, dtype=np.float64)

    # Load aggregation (instance grouping)
    agg_file = os.path.join(scene_dir, f"{scene_id}.aggregation.json")
    with open(agg_file, 'r') as f:
        agg_data = json.load(f)

    # Load segmentation (vertex to segment mapping)
    segs_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    with open(segs_file, 'r') as f:
        segs_data = json.load(f)
    seg_indices = np.array(segs_data['segIndices'])

    # Load mesh with coordinates and colors
    ply_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.labels.ply")
    plydata = PlyData.read(ply_file)
    vertices = plydata['vertex']

    # Extract coordinates (NOT aligned yet!)
    coords = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T

    # Apply axis alignment transformation
    coords_homog = np.hstack([coords, np.ones((coords.shape[0], 1), dtype=np.float64)])
    coords_aligned = (M_align @ coords_homog.T).T[:, :3]

    # Combine aligned coordinates with colors
    points = np.hstack([coords_aligned, colors])

    # Extract instances
    ins_pcds = []
    ins_labels = []
    ins_scores = []

    for seg_group in agg_data['segGroups']:
        label = seg_group['label']
        segments = seg_group['segments']

        mask = np.isin(seg_indices, segments)

        if np.any(mask):
            instance_points = points[mask]
            ins_pcds.append(instance_points)
            ins_labels.append(label)
            ins_scores.append(1.0)

    return ins_pcds, ins_labels, ins_scores


# ============================================================
# 6. Main Processing Function
# ============================================================

def process_gt_scene(
    scene_id,
    scans_root,
    color_root,
    save_root,
    min_pixels=120,
    min_ratio=0.1,
    max_points=200000
):
    """
    Process one GT scene and save in Mask3D format
    """
    # Check if already processed
    save_path = os.path.join(save_root, f"{scene_id}.npz")
    if os.path.exists(save_path):
        print(f"[SKIP] {scene_id}: already exists")
        return

    scene_dir = os.path.join(scans_root, scene_id)

    # 1. Load GT instances
    try:
        ins_pcds, ins_labels, ins_scores = load_scannet_gt_instances(scene_dir)
    except Exception as e:
        print(f"[ERROR] {scene_id}: Failed to load GT - {e}")
        return

    if len(ins_pcds) == 0:
        print(f"[WARNING] {scene_id}: No instances found")
        return

    # 2. Compute 3D bbox8
    bbox8_list = []
    for pts in ins_pcds:
        if len(pts) == 0:
            bbox8 = np.zeros((8, 3))
        else:
            bbox8 = compute_bbox8(pts)
        bbox8_list.append(bbox8)
    bbox8_array = np.stack(bbox8_list, axis=0)

    # 3. Read scene metadata
    try:
        M_align, intr = read_scene_txt_color_only(scene_dir)
    except Exception as e:
        print(f"[WARNING] {scene_id}: Failed to read metadata - {e}")
        # Save with 3D bbox only
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f"{scene_id}.npz")
        np.savez_compressed(
            save_path,
            ins_pcds=np.array(ins_pcds, dtype=object),
            ins_labels=np.array(ins_labels, dtype=object),
            ins_scores=np.array(ins_scores, dtype=np.float32),
            bbox8=bbox8_array
        )
        print(f"[✓] {scene_id}: Saved with bbox8 only → {save_path}")
        return

    # 4. List all available views
    color_dir = os.path.join(color_root, scene_id, "color")
    if not os.path.exists(color_dir):
        print(f"[WARNING] {scene_id}: Color directory not found: {color_dir}")
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f"{scene_id}.npz")
        np.savez_compressed(
            save_path,
            ins_pcds=np.array(ins_pcds, dtype=object),
            ins_labels=np.array(ins_labels, dtype=object),
            ins_scores=np.array(ins_scores, dtype=np.float32),
            bbox8=bbox8_array
        )
        print(f"[✓] {scene_id}: Saved with bbox8 only → {save_path}")
        return

    view_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    views = [os.path.splitext(os.path.basename(p))[0] for p in view_files]

    if not views:
        print(f"[WARNING] {scene_id}: No views found")
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f"{scene_id}.npz")
        np.savez_compressed(
            save_path,
            ins_pcds=np.array(ins_pcds, dtype=object),
            ins_labels=np.array(ins_labels, dtype=object),
            ins_scores=np.array(ins_scores, dtype=np.float32),
            bbox8=bbox8_array
        )
        print(f"[✓] {scene_id}: Saved with bbox8 only → {save_path}")
        return

    # 5. Project instances onto views
    views_per_instance = []
    bbox2d_per_instance = []
    ranging_ratio_per_instance = []
    vis_ratio_per_instance = []
    instances_per_view = {v: [] for v in views}

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

            depth_map = load_depth_map(depth_path)

            K = make_K_color_scaled(intr, W, H)
            T_c2w_aligned = M_align @ T_c2w
            T_w2c = np.linalg.inv(T_c2w_aligned)

            result = project_points_and_stats_color(
                K, T_w2c, pts3d, W, H, depth_map=depth_map,
                min_z=0.05, max_z=80.0
            )

            mask, rect, ranging_ratio, vis_ratio, ranging_cnt, vis_cnt, total_cnt = result

            if mask is None or rect is None:
                continue
            if vis_cnt < min_pixels or vis_ratio < min_ratio:
                continue

            pos_views.append(v)
            bbox_per_view[v] = [float(f"{rect[0]:.2f}"), float(f"{rect[1]:.2f}"),
                               float(f"{rect[2]:.2f}"), float(f"{rect[3]:.2f}")]
            ranging_ratio_per_view[v] = float(f"{ranging_ratio:.6f}")
            vis_ratio_per_view[v] = float(f"{vis_ratio:.6f}")
            instances_per_view[v].append(inst_idx)

        views_per_instance.append(sorted(pos_views, key=lambda x: int(x) if x.isdigit() else x))
        bbox2d_per_instance.append(dict(sorted(bbox_per_view.items(),
                                              key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])))
        ranging_ratio_per_instance.append(dict(sorted(ranging_ratio_per_view.items(),
                                                     key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])))
        vis_ratio_per_instance.append(dict(sorted(vis_ratio_per_view.items(),
                                                  key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])))

    # 6. Sort instances_per_view
    instances_per_view = dict(sorted(instances_per_view.items(),
                                    key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))

    # 7. Save
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, f"{scene_id}.npz")

    np.savez_compressed(
        save_path,
        ins_pcds=np.array(ins_pcds, dtype=object),
        ins_labels=np.array(ins_labels, dtype=object),
        ins_scores=np.array(ins_scores, dtype=np.float32),
        bbox8=bbox8_array,
        views_per_instance=np.array(views_per_instance, dtype=object),
        bbox2d_per_instance=np.array(bbox2d_per_instance, dtype=object),
        ranging_ratio_per_instance=np.array(ranging_ratio_per_instance, dtype=object),
        vis_ratio_per_instance=np.array(vis_ratio_per_instance, dtype=object),
        instances_per_view=np.array(instances_per_view, dtype=object)
    )

    print(f"[✓] {scene_id}: {len(ins_pcds)} GT instances → {save_path}")


# ============================================================
# 7. Parallel Processing Wrapper
# ============================================================

def process_scene_wrapper(scene_id, args_dict):
    """Wrapper for parallel processing"""
    try:
        process_gt_scene(
            scene_id=scene_id,
            scans_root=args_dict['scans_root'],
            color_root=args_dict['color_root'],
            save_root=args_dict['save_root'],
            min_pixels=args_dict['min_pixels'],
            min_ratio=args_dict['min_ratio'],
            max_points=args_dict['max_points']
        )
        return scene_id, True
    except Exception as e:
        print(f"[ERROR] {scene_id}: {e}")
        import traceback
        traceback.print_exc()
        return scene_id, False


# ============================================================
# 8. Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process ScanNet GT instances into Mask3D format"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE,
                        help="Path to config YAML file (default: base.yaml)")
    parser.add_argument("--splits", default="train,val",
                        help="Dataset splits to process (comma-separated, default: train,val)")
    parser.add_argument("--save_root", default=None,
                        help="Output directory (default: from config mask3d_pred_path/../GT)")
    parser.add_argument("--min_pixels", type=int, default=120,
                        help="Minimum visible pixels per view")
    parser.add_argument("--min_ratio", type=float, default=0.1,
                        help="Minimum visibility ratio per view")
    parser.add_argument("--max_points", type=int, default=200000,
                        help="Max points per instance")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract paths from config
    scans_root = cfg["raw_scene_path"]
    color_root = cfg["color_root"]
    metadata_root = cfg["scene_metadata"]["metadata_root_path"]

    # Determine save_root
    if args.save_root:
        save_root = args.save_root
    else:
        # Default: use gt_instance_path from config
        save_root = cfg.get("gt_instance_path", os.path.join(os.path.dirname(cfg.get("mask3d_pred_path", "")), "GT"))

    print(f"Loaded config from: {args.config}")
    print(f"  scans_root: {scans_root}")
    print(f"  color_root: {color_root}")
    print(f"  save_root: {save_root}")

    # Process each split
    splits = [s.strip() for s in args.splits.split(",")]
    for split in splits:
        scene_list_file = os.path.join(metadata_root, cfg["scene_metadata"][f"{split}_scene_ids"])
        print(f"\n==> Processing {split} split: {scene_list_file}")

        # Read scene list
        with open(scene_list_file, 'r') as f:
            scene_ids = [line.strip() for line in f if line.strip()]

        args_dict = {
            'scans_root': scans_root,
            'color_root': color_root,
            'save_root': save_root,
            'min_pixels': args.min_pixels,
            'min_ratio': args.min_ratio,
            'max_points': args.max_points
        }

        print(f"Processing {len(scene_ids)} GT scenes with {args.num_workers} worker(s)...")

        if args.num_workers > 1:
            print(f"[INFO] Using multiprocessing with {args.num_workers} workers")
            worker_func = partial(process_scene_wrapper, args_dict=args_dict)

            with Pool(processes=args.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_func, scene_ids),
                    total=len(scene_ids),
                    desc=f"Processing {split} GT scenes"
                ))

            success_count = sum(1 for _, success in results if success)
            failed_count = len(results) - success_count
            print(f"[OK] {split}: {success_count} succeeded, {failed_count} failed")
            if failed_count > 0:
                failed_scenes = [scene_id for scene_id, success in results if not success]
                print(f"[WARNING] Failed scenes: {failed_scenes}")
        else:
            print(f"[INFO] Using sequential processing")
            for scene_id in tqdm(scene_ids, desc=f"Processing {split} GT scenes"):
                process_gt_scene(
                    scene_id=scene_id,
                    scans_root=scans_root,
                    color_root=color_root,
                    save_root=save_root,
                    min_pixels=args.min_pixels,
                    min_ratio=args.min_ratio,
                    max_points=args.max_points
                )

            print(f"[OK] {split} split complete!")

    print(f"\n[OK] All splits processed!")


if __name__ == "__main__":
    main()
