#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess NR3D CSV: add evaluation labels and 3D bounding boxes.

This script merges functionality from:
1. M3DRef-CLIP/dataset/nr3d/add_evaluation_labels.py - adds is_easy, is_view_dep fields
2.  - adds ann_id, bbox_3d fields

Input: Raw NR3D CSV files (from data_dir in nr3d.yaml)
Output: Processed CSV with is_easy, is_view_dep, ann_id, bbox_3d fields (saved in-place, raw backed up)

All paths are loaded from config files:
- base.yaml: scene_dataset_path
- nr3d.yaml: data_dir, data_files
"""

import os
import csv
import json
import shutil
import argparse
import ast
import numpy as np
import torch
from tqdm import tqdm

from llava.dataset.config import load_dataset_config


# ============================================================
# Step 1: Add evaluation labels (is_easy, is_view_dep)
# From M3DRef-CLIP add_evaluation_labels.py
# ============================================================

def decode_stimulus_string(s):
    """
    Decode stimulus_id string to extract scene info.

    Format: scene_id-instance_label-n_objects-target_id[-distractors_ids]

    Returns:
        tuple: (scene_id, instance_label, n_objects, target_id, distractors_ids)
    """
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = s.split('-', maxsplit=4)

    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1
    return scene_id, instance_label, n_objects, target_id, distractors_ids


def is_easy(stimulus_id):
    """Check if utterance is 'easy' (n_objects <= 2)."""
    return decode_stimulus_string(stimulus_id)[2] <= 2


def is_view_dependent(tokens_str):
    """Check if utterance contains view-dependent words."""
    target_words = {
        'front', 'behind', 'back', 'right', 'left', 'facing',
        'leftmost', 'rightmost', 'looking', 'across'
    }
    tokens = ast.literal_eval(tokens_str)
    return len(set(tokens).intersection(target_words)) > 0


# ============================================================
# Step 2: Add ann_id and 3D bounding boxes
# From nr3d_update_info.py
# ============================================================

def get_bbox_from_pth(pth_root, split, scene_id, object_id, pth_cache=None):
    """
    Load bbox corners for a specific object from .pth file.

    Args:
        pth_root: Root directory containing .pth files
        split: Dataset split (train/test -> train/val for pth)
        scene_id: Scene ID (e.g., 'scene0000_00')
        object_id: Object ID to look up
        pth_cache: Optional dict to cache loaded pth data

    Returns:
        numpy.ndarray: 8x3 corner array, or None if not found
    """
    if pth_cache is None:
        pth_cache = {}

    # Map NR3D split to pth split (test -> val)
    pth_split = "val" if split == "test" else split

    cache_key = (pth_split, scene_id)

    # Load pth data if not cached
    if cache_key not in pth_cache:
        pth_path = os.path.join(pth_root, pth_split, f"{scene_id}.pth")
        if not os.path.exists(pth_path):
            return None

        data = torch.load(pth_path, map_location="cpu")
        aabb_ids = np.asarray(data.get("aabb_obj_ids"))
        aabb_xyz = np.asarray(data.get("aabb_corner_xyz"))

        if aabb_ids is None or aabb_xyz is None:
            return None

        pth_cache[cache_key] = (aabb_ids, aabb_xyz)

    aabb_ids, aabb_xyz = pth_cache[cache_key]

    # Find object by ID
    idx = np.where(aabb_ids == int(object_id))[0]
    if idx.size == 0:
        return None

    return aabb_xyz[idx[0]].astype(np.float64)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NR3D: add is_easy, is_view_dep, ann_id, bbox_3d fields"
    )
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Dataset split to process")
    args = parser.parse_args()

    # Load config (merges base.yaml + nr3d.yaml)
    cfg = load_dataset_config("nr3d")

    # Extract paths from config
    data_dir = cfg["data_dir"]
    data_files = cfg["data_files"]
    pth_root = cfg["scene_dataset_path"]

    print(f"Configuration (from YAML):")
    print(f"  Data dir: {data_dir}")
    print(f"  PTH root: {pth_root}")

    # Get CSV file for the specified split
    csv_filename = data_files.get(args.split)
    if not csv_filename:
        print(f"Error: No CSV file configured for split '{args.split}'")
        return

    csv_file = os.path.join(data_dir, csv_filename)
    print(f"\nProcessing: {csv_file}")

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return

    # Backup raw file before processing
    backup_dir = os.path.join(data_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, csv_filename)

    if not os.path.exists(backup_path):
        shutil.copy(csv_file, backup_path)
        print(f"[BACKUP] Raw file copied to {backup_path}")
    else:
        print(f"[BACKUP] Backup already exists at {backup_path}")

    # Read all rows
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows")

    # Check if already processed
    if 'is_easy' in fieldnames and 'is_view_dep' in fieldnames and 'ann_id' in fieldnames and 'bbox_3d' in fieldnames:
        print(f"[SKIP] File already processed (has is_easy, is_view_dep, ann_id, bbox_3d)")
        return

    # Step 1: Add is_easy and is_view_dep
    print("\n[Step 1] Adding is_easy and is_view_dep labels...")
    easy_count = 0
    view_dep_count = 0

    for row in tqdm(rows, desc="Adding eval labels"):
        row['is_easy'] = is_easy(row['stimulus_id'])
        row['is_view_dep'] = is_view_dependent(row['tokens'])
        if row['is_easy']:
            easy_count += 1
        if row['is_view_dep']:
            view_dep_count += 1

    print(f"  is_easy distribution: easy={easy_count}, hard={len(rows) - easy_count}")
    print(f"  is_view_dep distribution: view_dep={view_dep_count}, view_indep={len(rows) - view_dep_count}")

    # Step 2: Add ann_id and bbox_3d
    print("\n[Step 2] Adding ann_id and bbox_3d...")
    tmp_ann_id_count = {}
    pth_cache = {}
    skipped_bbox = 0

    for row in tqdm(rows, desc="Adding ann_id and bbox_3d"):
        scan_id = row['scan_id']
        target_id = int(row['target_id'])

        # Generate ann_id
        scene_obj_key = (scan_id, target_id)
        if scene_obj_key not in tmp_ann_id_count:
            tmp_ann_id_count[scene_obj_key] = 0
        else:
            tmp_ann_id_count[scene_obj_key] += 1
        row['ann_id'] = tmp_ann_id_count[scene_obj_key]

        # Get bbox from pth
        bbox = get_bbox_from_pth(pth_root, args.split, scan_id, target_id, pth_cache)
        if bbox is not None:
            row['bbox_3d'] = json.dumps(bbox.tolist())
        else:
            row['bbox_3d'] = ''
            skipped_bbox += 1

    # Build new fieldnames with added columns
    new_fieldnames = ['ann_id'] + fieldnames + ['is_easy', 'is_view_dep', 'bbox_3d']
    # Remove duplicates while preserving order
    seen = set()
    new_fieldnames = [x for x in new_fieldnames if not (x in seen or seen.add(x))]

    # Save processed file (overwrites raw, which was already backed up)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Saved {len(rows)} rows to {csv_file}")
    print(f"  Added columns: ann_id, is_easy, is_view_dep, bbox_3d")
    if skipped_bbox > 0:
        print(f"  Rows without bbox: {skipped_bbox}")

    # Print ann_id statistics
    unique_pairs = len(tmp_ann_id_count)
    print(f"\n  Statistics:")
    print(f"    Total utterances: {len(rows)}")
    print(f"    Unique (scan_id, target_id) pairs: {unique_pairs}")
    print(f"    Average utterances per object: {len(rows)/unique_pairs:.2f}")


if __name__ == "__main__":
    main()
