#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess ScanRefer JSON: add evaluation labels (unique/multiple) and 3D bounding boxes.

This script merges functionality from:
1. M3DRef-CLIP/dataset/scanrefer/add_evaluation_labels.py - adds eval_type field
2. scanrefer_update_info.py - adds bbox_3d field

Input: Raw ScanRefer JSON files (from data_dir in scanrefer.yaml)
Output: Processed JSON with eval_type and bbox_3d fields (saved in-place, raw backed up)

All paths are loaded from config files:
- base.yaml: scene_dataset_path, scene_metadata (label_mapping_file, valid_semantic_mapping)
- scanrefer.yaml: data_dir, data_files
"""

import os
import csv
import json
import shutil
import argparse
import numpy as np
import torch
from tqdm import tqdm

from llava.dataset.config import load_dataset_config, load_base_config


# ============================================================
# Step 1: Add evaluation labels (unique/multiple)
# From M3DRef-CLIP add_evaluation_labels.py
# ============================================================

def get_semantic_mapping_file(file_path, mapping_name="nyu40"):
    """
    Load semantic label mapping from TSV file.

    Args:
        file_path: Path to scannetv2-labels.combined.tsv
        mapping_name: One of "nyu40", "eigen13", "mpcat40"

    Returns:
        dict: Maps object name (str) -> semantic label (int)
    """
    label_mapping = {}
    mapping_col_idx = {
        "nyu40": 4,
        "eigen13": 5,
        "mpcat40": 16
    }
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[mapping_col_idx[mapping_name]])
    return label_mapping


def add_eval_type_labels(items, label_mapping, valid_semantic_mapping):
    """
    Add eval_type (unique/multiple) to each item based on semantic class count per scene.

    Logic:
    - Count unique objects per (scene_id, semantic_label) pair
    - If only one object of that semantic class in the scene -> "unique"
    - Otherwise -> "multiple"

    Args:
        items: List of ScanRefer items
        label_mapping: Object name -> semantic label mapping
        valid_semantic_mapping: List of valid semantic labels

    Returns:
        List of items with eval_type field added
    """
    # First pass: count unique objects per (scene, semantic_label)
    obj_cache = {}
    sem_cache = {}

    for item in items:
        if (item["scene_id"], item["object_id"]) in obj_cache:
            continue

        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39  # default: "otherprop"
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39

        if (item['scene_id'], sem_label) not in sem_cache:
            sem_cache[(item['scene_id'], sem_label)] = 0
        sem_cache[(item['scene_id'], sem_label)] += 1
        obj_cache[(item["scene_id"], item["object_id"])] = True

    # Second pass: assign eval_type
    for item in items:
        scene_id = item['scene_id']
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39

        assert sem_cache[(scene_id, sem_label)] >= 1
        item["eval_type"] = "unique" if sem_cache[(scene_id, sem_label)] == 1 else "multiple"

    return items


# ============================================================
# Step 2: Add 3D bounding boxes
# From scanrefer_update_info.py
# ============================================================

def get_bbox_from_pth(pth_root, split, scene_id, oid):
    """
    Load bbox corners for a specific object from .pth file.

    Args:
        pth_root: Root directory containing split folders with .pth files
        split: "train" or "val"
        scene_id: Scene ID (e.g., "scene0000_00")
        oid: Object ID (int)

    Returns:
        numpy.ndarray: 8x3 corner array, or None if not found
    """
    pth_path = os.path.join(pth_root, split, f"{scene_id}.pth")
    if not os.path.exists(pth_path):
        return None

    data = torch.load(pth_path, map_location="cpu")
    aabb_ids = np.asarray(data.get("aabb_obj_ids"))
    aabb_xyz = np.asarray(data.get("aabb_corner_xyz"))

    if aabb_ids is None or aabb_xyz is None:
        return None

    idx = np.where(aabb_ids == int(oid))[0]
    if idx.size == 0:
        return None

    return aabb_xyz[idx[0]].astype(np.float64)


def add_bbox_3d(items, pth_root, split):
    """
    Add bbox_3d field to each item.

    Args:
        items: List of ScanRefer items
        pth_root: Root directory for .pth files
        split: "train" or "val"

    Returns:
        Tuple of (updated_items, skipped_count)
    """
    updated_items = []
    skipped = 0

    for item in tqdm(items, desc=f"Adding bbox_3d ({split})"):
        scene_id = item["scene_id"]
        object_id = int(item["object_id"])

        corners = get_bbox_from_pth(pth_root, split, scene_id, object_id)

        if corners is None:
            skipped += 1
            continue

        item["bbox_3d"] = corners.tolist()
        updated_items.append(item)

    return updated_items, skipped


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ScanRefer: add eval_type and bbox_3d fields"
    )
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Dataset split to process")
    args = parser.parse_args()

    # Load config (merges base.yaml + scanrefer.yaml)
    cfg = load_dataset_config("scanrefer")
    base_cfg = load_base_config()

    # Extract paths from config
    data_dir = cfg["data_dir"]
    pth_root = cfg["scene_dataset_path"]

    # Label mapping file path from base config
    metadata_root = base_cfg["scene_metadata"]["metadata_root_path"]
    label_mapping_filename = base_cfg["scene_metadata"]["label_mapping_file"]
    label_mapping_file = os.path.join(metadata_root, label_mapping_filename)

    # Valid semantic mapping from base config
    valid_semantic_mapping = base_cfg["scene_metadata"]["valid_semantic_mapping"]

    print(f"Configuration (from YAML):")
    print(f"  Data dir: {data_dir}")
    print(f"  PTH root: {pth_root}")
    print(f"  Label mapping: {label_mapping_file}")
    print(f"  Valid semantic labels: {valid_semantic_mapping}")

    # Input/output file path (same - will backup raw first)
    json_file = os.path.join(data_dir, f"ScanRefer_filtered_{args.split}.json")
    print(f"\nProcessing: {json_file}")

    # Backup raw file before processing
    backup_dir = os.path.join(data_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"ScanRefer_filtered_{args.split}.json")

    if not os.path.exists(backup_path):
        shutil.copy(json_file, backup_path)
        print(f"[BACKUP] Raw file copied to {backup_path}")
    else:
        print(f"[BACKUP] Backup already exists at {backup_path}")

    # Load raw data
    with open(json_file, "r") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items")

    # Check if already processed
    if items and "bbox_3d" in items[0] and "eval_type" in items[0]:
        print(f"[SKIP] File already processed (has bbox_3d and eval_type)")
        return

    # Step 1: Add eval_type labels
    print("\n[Step 1] Adding eval_type labels...")
    label_mapping = get_semantic_mapping_file(label_mapping_file, "nyu40")
    items = add_eval_type_labels(items, label_mapping, valid_semantic_mapping)

    unique_count = sum(1 for item in items if item.get("eval_type") == "unique")
    multiple_count = sum(1 for item in items if item.get("eval_type") == "multiple")
    print(f"  eval_type distribution: unique={unique_count}, multiple={multiple_count}")

    # Step 2: Add bbox_3d
    print("\n[Step 2] Adding bbox_3d from .pth files...")
    items, skipped = add_bbox_3d(items, pth_root, args.split)

    # Save processed file (overwrites raw, which was already backed up)
    with open(json_file, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved {len(items)} items to {json_file}")
    if skipped > 0:
        print(f"  Skipped {skipped} items (bbox not found in .pth files)")


if __name__ == "__main__":
    main()
