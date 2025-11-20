#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate NR3D predictions with multiple random runs.
Runs N times with random bbox selection and reports mean/std.
"""

import argparse
import csv
import json
import os
import pickle
import numpy as np
import torch
import yaml
from tqdm import tqdm


def get_batch_aabb_pair_ious(batch_boxes_1_bound, batch_boxes_2_bound):
    """Compute IoU between pairs of 3D bounding boxes"""
    box_1_x_min, box_1_y_min, box_1_z_min = torch.tensor_split(batch_boxes_1_bound[:, 0], 3, dim=1)
    box_1_x_max, box_1_y_max, box_1_z_max = torch.tensor_split(batch_boxes_1_bound[:, 1], 3, dim=1)

    box_2_x_min, box_2_y_min, box_2_z_min = torch.tensor_split(batch_boxes_2_bound[:, 0], 3, dim=1)
    box_2_x_max, box_2_y_max, box_2_z_max = torch.tensor_split(batch_boxes_2_bound[:, 1], 3, dim=1)

    x_a = torch.maximum(box_1_x_min, box_2_x_min)
    y_a = torch.maximum(box_1_y_min, box_2_y_min)
    z_a = torch.maximum(box_1_z_min, box_2_z_min)
    x_b = torch.minimum(box_1_x_max, box_2_x_max)
    y_b = torch.minimum(box_1_y_max, box_2_y_max)
    z_b = torch.minimum(box_1_z_max, box_2_z_max)

    zero_tensor = torch.zeros_like(x_a)
    intersection_volume = torch.maximum((x_b - x_a), zero_tensor) * torch.maximum((y_b - y_a), zero_tensor) * \
                          torch.maximum((z_b - z_a), zero_tensor)
    box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
    box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)
    iou = intersection_volume / (box_1_volume + box_2_volume - intersection_volume + torch.finfo(torch.float32).eps)
    return iou.flatten()


# NR3D uses GT boxes as candidates, so IoU threshold is high
IOU_THRESHOLD = 0.9


class ReferIt3DEvaluator:
    """Evaluator for NR3D dataset following M3DRef-CLIP's ReferIt3DEvaluator."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.ground_truths = None
        self.evaluation_types = {"easy_dep": 0, "easy_indep": 1, "hard_dep": 2, "hard_indep": 3}
        self.evaluation_types_comb = {"easy": (0, 1), "hard": (2, 3), "view_dep": (0, 2), "view_indep": (1, 3)}

    def set_ground_truths(self, ground_truths):
        """Set ground truths."""
        self.ground_truths = ground_truths

    def _print_results(self, results, std_results=None):
        """Print results with optional std."""
        print(f"{'=' * 67}")
        print("{0:<12}{1:<12}{2:<12}{3:<12}{4:<12}".format("", "easy", "hard", "view-dep", "view-indep", "overall"))
        print(f"{'-' * 67}")
        line_str = '{:<12}'.format("Acc")
        for sub_group_type in ['easy', 'hard', 'view_dep', 'view_indep', 'overall']:
            score = results[sub_group_type]
            if std_results:
                std = std_results[sub_group_type]
                line_str += '{:<12}'.format(f"{score*100:.2f}±{std*100:.2f}")
            else:
                line_str += '{:<12.2f}'.format(score * 100)
        print(line_str)
        print(f"{'=' * 67}")

    def evaluate(self, predictions):
        """Evaluate predictions on ground truths."""
        all_gt_info_len = len(self.ground_truths)
        eval_type_mask = np.empty(all_gt_info_len, dtype=np.uint8)
        tps = np.zeros(all_gt_info_len, dtype=bool)

        gt_keys = list(self.ground_truths.keys())

        for i, key in enumerate(gt_keys):
            gt_info = self.ground_truths[key]
            eval_type_mask[i] = self.evaluation_types[gt_info["eval_type"]]

            if key in predictions:
                pred_info = predictions[key]
                iou = get_batch_aabb_pair_ious(
                    torch.from_numpy(pred_info["aabb_bound"]),
                    torch.from_numpy(gt_info["aabb_bound"])
                )[0].item()
                if iou >= IOU_THRESHOLD:
                    tps[i] = True

        # Compute results by evaluation type
        results = {}
        for sub_group in self.evaluation_types_comb.keys():
            selected_indices = np.isin(eval_type_mask, np.array(self.evaluation_types_comb[sub_group], dtype=np.uint8))
            if np.any(selected_indices):
                results[sub_group] = np.count_nonzero(tps[selected_indices]) / np.count_nonzero(selected_indices)
            else:
                results[sub_group] = np.nan
        results["overall"] = np.count_nonzero(tps) / tps.shape[0]

        return results


def load_ground_truths(gt_csv_path, npz_dir):
    """Load ground truth data from NR3D CSV and NPZ files."""
    print(f"Loading ground truths from {gt_csv_path}")

    ground_truths = {}
    scene_bbox_cache = {}
    tmp_ann_id_count = {}

    with open(gt_csv_path, 'r') as f:
        csv_data = csv.DictReader(f)

        for row in csv_data:
            scene_id = row['scan_id']
            object_id = int(row['target_id'])

            # Generate ann_id
            scene_obj_key = (scene_id, object_id)
            if scene_obj_key not in tmp_ann_id_count:
                tmp_ann_id_count[scene_obj_key] = 0
            else:
                tmp_ann_id_count[scene_obj_key] += 1
            ann_id = tmp_ann_id_count[scene_obj_key]

            # Determine eval_type
            is_easy = row['is_easy'] == 'True'
            is_view_dep = row['is_view_dep'] == 'True'

            if is_easy and is_view_dep:
                eval_type = "easy_dep"
            elif is_easy:
                eval_type = "easy_indep"
            elif is_view_dep:
                eval_type = "hard_dep"
            else:
                eval_type = "hard_indep"

            # Load scene NPZ data if not cached
            if scene_id not in scene_bbox_cache:
                npz_path = os.path.join(npz_dir, f"{scene_id}.npz")
                if not os.path.exists(npz_path):
                    continue

                npz_data = np.load(npz_path, allow_pickle=True)
                bbox8_all = npz_data['bbox8']

                aabb_bounds_dict = {}
                for inst_idx in range(len(bbox8_all)):
                    bbox8 = bbox8_all[inst_idx]
                    min_point = bbox8.min(axis=0)
                    max_point = bbox8.max(axis=0)
                    aabb = np.array([[min_point, max_point]], dtype=np.float32)
                    aabb_bounds_dict[inst_idx] = aabb

                scene_bbox_cache[scene_id] = aabb_bounds_dict

            if object_id not in scene_bbox_cache.get(scene_id, {}):
                continue

            aabb_bound = scene_bbox_cache[scene_id][object_id]

            key = (scene_id, str(object_id), ann_id)
            ground_truths[key] = {
                'aabb_bound': aabb_bound,
                'eval_type': eval_type
            }

    print(f"Loaded {len(ground_truths)} ground truth samples")
    return ground_truths


def load_predictions(pred_pkl_path):
    """Load predictions from pickle file."""
    print(f"Loading predictions from {pred_pkl_path}")
    with open(pred_pkl_path, 'rb') as f:
        predictions = pickle.load(f)
    print(f"Loaded {len(predictions)} predictions")
    return predictions


def load_config(config_path):
    """Load dataset config from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_predictions_random(predictions, ground_truths, rng):
    """Process predictions with random bbox selection."""
    processed_predictions = {}

    for pred_key, pred_info in predictions.items():
        if pred_key not in ground_truths:
            continue

        aabb_bound = pred_info['aabb_bound']
        if not isinstance(aabb_bound, np.ndarray):
            aabb_bound = np.array(aabb_bound, dtype=np.float32)

        if aabb_bound.ndim == 3:
            if aabb_bound.shape[0] > 1:
                selected_idx = rng.integers(0, aabb_bound.shape[0])
                selected_box = aabb_bound[selected_idx:selected_idx+1]
            else:
                selected_box = aabb_bound
        elif aabb_bound.ndim == 2 and aabb_bound.shape == (2, 3):
            selected_box = np.expand_dims(aabb_bound, axis=0)
        else:
            continue

        processed_predictions[pred_key] = {'aabb_bound': selected_box}

    return processed_predictions


def main(args):
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'nr3d.yaml')
    config = load_config(config_path)

    # Determine paths from config
    data_dir = config['data_dir']
    llava_grounding_dir = config['llava_grounding_dir']
    verification_file = config['verification_file']
    test_file = config['data_files']['test']

    # Ground truth path
    gt_csv_path = args.gt_csv_path or os.path.join(data_dir, test_file)

    # NPZ directory for GT boxes
    npz_dir = args.npz_dir or config.get("gt_instance_path", "")

    # Prediction path
    pred_pkl_path = args.pred_pkl_path or os.path.join(
        llava_grounding_dir,
        verification_file.replace('.json', '_predictions.pkl')
    )

    print(f"\n{'=' * 60}")
    print(f"NR3D Evaluation ({args.num_runs} random runs)")
    print(f"{'=' * 60}")
    print(f"Ground truth: {gt_csv_path}")
    print(f"NPZ dir:      {npz_dir}")
    print(f"Predictions:  {pred_pkl_path}")
    print(f"{'=' * 60}\n")

    # Load data
    ground_truths = load_ground_truths(gt_csv_path, npz_dir)
    predictions = load_predictions(pred_pkl_path)

    # Statistics
    num_predictions = len(predictions)
    num_gt = len(ground_truths)
    print(f"\nEvaluation Statistics:")
    print(f"  Ground truth samples:  {num_gt}")
    print(f"  Predictions:           {num_predictions} ({num_predictions/num_gt*100:.2f}%)")
    print(f"  Missing:               {num_gt - num_predictions} ({(num_gt-num_predictions)/num_gt*100:.2f}%)")

    # Run N times with random selection
    all_results = {key: [] for key in ['easy', 'hard', 'view_dep', 'view_indep', 'overall']}

    evaluator = ReferIt3DEvaluator(verbose=False)
    evaluator.set_ground_truths(ground_truths)

    print(f"\nRunning {args.num_runs} random evaluations...")
    for run_idx in tqdm(range(args.num_runs), desc="Random runs"):
        rng = np.random.default_rng(seed=100 + run_idx)
        processed_predictions = process_predictions_random(predictions, ground_truths, rng)
        results = evaluator.evaluate(processed_predictions)

        for key in all_results:
            all_results[key].append(results[key])

    # Compute mean and std
    mean_results = {key: np.mean(values) for key, values in all_results.items()}
    std_results = {key: np.std(values) for key, values in all_results.items()}

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Results (mean ± std over {args.num_runs} runs)")
    print(f"{'=' * 80}")
    print("{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format("easy", "hard", "view-dep", "view-indep", "overall"))
    print(f"{'-' * 80}")
    line_str = ''
    for key in ['easy', 'hard', 'view_dep', 'view_indep', 'overall']:
        mean = mean_results[key] * 100
        std = std_results[key] * 100
        line_str += '{:<15}'.format(f"{mean:.2f}±{std:.2f}")
    print(line_str)
    print(f"{'=' * 80}\n")

    # Save results if output file specified
    if args.output_file:
        output_data = {
            'method': args.method_name,
            'num_runs': args.num_runs,
            'num_predictions': num_predictions,
            'num_ground_truths': num_gt,
            'coverage': num_predictions / num_gt,
            'mean': {key: float(mean_results[key]) for key in mean_results},
            'std': {key: float(std_results[key]) for key in std_results},
            'all_runs': {key: [float(v) for v in values] for key, values in all_results.items()}
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"[SAVE] Results saved to {args.output_file}")

    return mean_results, std_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NR3D predictions with random runs")
    parser.add_argument(
        "--pred-pkl-path",
        type=str,
        default=None,
        help="Path to predictions pickle file (default: from config)"
    )
    parser.add_argument(
        "--gt-csv-path",
        type=str,
        default=None,
        help="Path to ground truth CSV file (default: from config)"
    )
    parser.add_argument(
        "--npz-dir",
        type=str,
        default=None,
        help="Directory containing GT NPZ files"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of random runs (default: 50)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default="LLaVA-NR3D",
        help="Name of the method being evaluated"
    )

    args = parser.parse_args()
    main(args)
