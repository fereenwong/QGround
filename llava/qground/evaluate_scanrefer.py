#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ScanRefer predictions.
Simplified evaluation script that directly evaluates predictions without multiple strategies.
"""

import argparse
import json
import os
import pickle
import numpy as np
import torch
import yaml
from llava.dataset.config import load_dataset_config
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


class ScanReferEvaluator:
    """Evaluator for ScanRefer 3D grounding task."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.ground_truths = None
        self.evaluation_types = {"unique": 0, "multiple": 1}

    def set_ground_truths(self, ground_truths):
        """Set ground truths."""
        self.ground_truths = ground_truths

    def _print_results(self, iou_25_results, iou_50_results):
        print(f"{'=' * 47}")
        print("{0:<12}{1:<12}{2:<12}{3:<12}".format("IoU", "unique", "multiple", "overall"))
        print(f"{'-' * 47}")
        line_1_str = '{:<12}'.format("0.25")
        for sub_group_type, score in iou_25_results.items():
            line_1_str += '{:<12.2f}'.format(score * 100)
        print(line_1_str)
        line_2_str = '{:<12}'.format("0.50")
        for sub_group_type, score in iou_50_results.items():
            line_2_str += '{:<12.2f}'.format(score * 100)
        print(line_2_str)
        print(f"{'=' * 47}")

    def _evaluate_one_query(self, pred_info, gt_info):
        """Evaluate one query"""
        iou_25_tp = 0
        iou_50_tp = 0

        iou = get_batch_aabb_pair_ious(
            torch.from_numpy(pred_info["aabb_bound"]),
            torch.from_numpy(gt_info["aabb_bound"])
        )[0].item()

        if iou >= 0.25:
            iou_25_tp += 1
        if iou >= 0.5:
            iou_50_tp += 1

        return iou_25_tp, iou_50_tp

    def evaluate(self, predictions):
        """Evaluate predictions on ground truths."""
        all_gt_info_len = len(self.ground_truths)
        eval_type_mask = np.empty(all_gt_info_len, dtype=bool)
        iou_25_tps = np.zeros(all_gt_info_len, dtype=bool)
        iou_50_tps = np.zeros(all_gt_info_len, dtype=bool)

        gt_keys = list(self.ground_truths.keys())

        num_predictions = len(predictions)
        num_missing = all_gt_info_len - num_predictions

        if self.verbose:
            print(f"\nEvaluation Statistics:")
            print(f"  Ground truth samples:  {all_gt_info_len}")
            print(f"  Predictions:           {num_predictions} ({num_predictions/all_gt_info_len*100:.1f}%)")
            print(f"  Missing:               {num_missing} ({num_missing/all_gt_info_len*100:.1f}%)")

        iterator = enumerate(tqdm(gt_keys, desc="Evaluating") if self.verbose else gt_keys)

        for i, key in iterator:
            gt_info = self.ground_truths[key]
            eval_type_mask[i] = self.evaluation_types[gt_info["eval_type"]]

            if key in predictions:
                pred_info = predictions[key]
                iou_25_tps[i], iou_50_tps[i] = self._evaluate_one_query(pred_info, gt_info)

        # Compute results by evaluation type
        iou_25_results = {}
        iou_50_results = {}

        for sub_group in self.evaluation_types.keys():
            selected_indices = eval_type_mask == self.evaluation_types[sub_group]
            if np.any(selected_indices):
                iou_25_results[sub_group] = np.count_nonzero(iou_25_tps[selected_indices]) / np.count_nonzero(selected_indices)
                iou_50_results[sub_group] = np.count_nonzero(iou_50_tps[selected_indices]) / np.count_nonzero(selected_indices)
            else:
                iou_25_results[sub_group] = np.nan
                iou_50_results[sub_group] = np.nan

        iou_25_results["overall"] = np.count_nonzero(iou_25_tps) / iou_25_tps.shape[0]
        iou_50_results["overall"] = np.count_nonzero(iou_50_tps) / iou_50_tps.shape[0]

        if self.verbose:
            self._print_results(iou_25_results, iou_50_results)

        return {"Acc@0.25": iou_25_results, "Acc@0.5": iou_50_results}


def load_ground_truths(gt_json_path):
    """Load ground truth data"""
    print(f"Loading ground truths from {gt_json_path}")
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    ground_truths = {}

    for item in gt_data:
        scene_id = item['scene_id']
        object_id = str(item['object_id'])  # Ensure string
        ann_id = str(item['ann_id'])  # Ensure string
        eval_type = item['eval_type']
        bbox_3d = np.array(item['bbox_3d'], dtype=np.float32)

        min_point = bbox_3d.min(axis=0)
        max_point = bbox_3d.max(axis=0)
        aabb_bound = np.array([[min_point, max_point]], dtype=np.float32)

        key = (scene_id, object_id, ann_id)
        ground_truths[key] = {
            'aabb_bound': aabb_bound,
            'eval_type': eval_type
        }

    print(f"Loaded {len(ground_truths)} ground truth samples")
    return ground_truths


def load_predictions(pred_pkl_path):
    """Load predictions from pickle file"""
    print(f"Loading predictions from {pred_pkl_path}")
    with open(pred_pkl_path, 'rb') as f:
        predictions = pickle.load(f)
    print(f"Loaded {len(predictions)} predictions")
    return predictions


def load_config(config_path):
    """Load dataset config from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'scanrefer.yaml')
    config = load_dataset_config("scanrefer")

    # Determine paths from config
    data_dir = config['data_dir']
    llava_grounding_dir = config['llava_grounding_dir']
    verification_file = config['verification_file']
    val_file = config['data_files']['val']

    # Ground truth path: data_dir + data_files['val']
    gt_json_path = args.gt_json_path or os.path.join(data_dir, val_file)

    # Prediction path: llava_grounding_dir + verification_file with .json -> _predictions.pkl
    pred_pkl_path = args.pred_pkl_path or os.path.join(
        llava_grounding_dir,
        verification_file.replace('.json', '_predictions.pkl')
    )

    print(f"\n{'=' * 60}")
    print(f"ScanRefer Evaluation")
    print(f"{'=' * 60}")
    print(f"Ground truth: {gt_json_path}")
    print(f"Predictions:  {pred_pkl_path}")
    print(f"{'=' * 60}\n")

    # Load data
    ground_truths = load_ground_truths(gt_json_path)
    predictions = load_predictions(pred_pkl_path)

    # Ensure predictions have correct format (single bbox per sample)
    processed_predictions = {}
    for key, pred_info in predictions.items():
        aabb_bound = pred_info['aabb_bound']
        if not isinstance(aabb_bound, np.ndarray):
            aabb_bound = np.array(aabb_bound, dtype=np.float32)

        # Ensure shape is (1, 2, 3)
        if aabb_bound.ndim == 3:
            aabb_bound = aabb_bound[0:1]  # Take first bbox if multiple
        elif aabb_bound.ndim == 2 and aabb_bound.shape == (2, 3):
            aabb_bound = np.expand_dims(aabb_bound, axis=0)

        processed_predictions[key] = {'aabb_bound': aabb_bound}

    # Evaluate
    evaluator = ScanReferEvaluator(verbose=True)
    evaluator.set_ground_truths(ground_truths)
    results = evaluator.evaluate(processed_predictions)

    # Save results if output file specified
    if args.output_file:
        output_data = {
            'method': args.method_name,
            'num_predictions': len(predictions),
            'num_ground_truths': len(ground_truths),
            'coverage': len(predictions) / len(ground_truths),
            'results': {
                metric_name: {
                    eval_type: float(score) if not np.isnan(score) else None
                    for eval_type, score in metric_results.items()
                }
                for metric_name, metric_results in results.items()
            }
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n[SAVE] Results saved to {args.output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ScanRefer predictions")
    parser.add_argument(
        "--pred-pkl-path",
        type=str,
        default=None,
        help="Path to predictions pickle file (default: from config)"
    )
    parser.add_argument(
        "--gt-json-path",
        type=str,
        default=None,
        help="Path to ground truth JSON file (default: from config)"
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
        default="LLaVA-ScanRefer",
        help="Name of the method being evaluated"
    )

    args = parser.parse_args()
    main(args)
