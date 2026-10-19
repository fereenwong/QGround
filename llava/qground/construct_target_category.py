#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construct target category candidates using LLaVA for semantic matching.

Unified script that handles both ScanRefer and NR3D datasets.

Given:
1. Parsed descriptions with target object classes
2. Instance predictions (Mask3D) or GT instances

Use LLaVA to determine which instance labels semantically match the target class,
handling cases like "cabinet" matching "kitchen cabinet", "wall cabinet", etc.

Usage:
    # For ScanRefer (uses Mask3D predictions)
    python construct_target_category.py --dataset scanrefer

    # For NR3D (uses GT instances)
    python construct_target_category.py --dataset nr3d

    # With custom paths
    python construct_target_category.py --dataset scanrefer --parsed-file /path/to/file.json
"""

import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.dataset.config import MODEL_PATH, Category_match_prompt, load_dataset_config
from typing import Dict
import transformers


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len: int = 2048,
    system_message: str = "You are a helpful assistant."
) -> torch.Tensor:
    """Preprocess conversation for Qwen model"""
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    # Apply prompt templates
    input_ids = []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id = []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system

    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id

    input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def query_llava_batch_semantic_match(
    model,
    tokenizer,
    conv_mode: str,
    target_class: str,
    instance_labels: list,
    temperature: float = 0.0
) -> Dict[str, bool]:
    """
    Query LLaVA to determine which instance labels semantically match the target class.
    Uses batching for efficiency - asks about all labels in a single forward pass.

    Args:
        model: LLaVA model
        tokenizer: tokenizer
        conv_mode: conversation mode
        target_class: target object class (e.g., "cabinet")
        instance_labels: list of instance labels (e.g., ["chair", "cabinet", "kitchen cabinet"])
        temperature: sampling temperature

    Returns:
        dict: {label: bool} mapping of each label to whether it matches
    """
    # Get system prompt from config
    system_prompt = Category_match_prompt

    # Create batch query
    labels_str = ", ".join([f'"{label}"' for label in instance_labels])
    query = f"""Target class: "{target_class}"
Instance labels: [{labels_str}]
For EACH label, answer YES or NO.
Respond in JSON format: {{"label1": "YES", "label2": "NO", ...}}"""

    try:
        conv = conv_templates[conv_mode].copy()
        line = {'from': 'human', 'value': query}

        input_ids = preprocess_qwen(
            [line, {'from': 'gpt', 'value': None}],
            tokenizer,
            has_image=False,
            system_message=system_prompt
        ).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]

        # Parse JSON response - handle both raw JSON and ```json blocks
        json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', outputs)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', outputs)
            json_str = json_match.group(0) if json_match else None

        if json_str:
            result_dict = json.loads(json_str)

            # Convert to boolean dict
            matches = {}
            for label in instance_labels:
                label_str = str(label)
                if label_str in result_dict:
                    answer = result_dict[label_str].strip().upper()
                    matches[label_str] = "YES" in answer
                else:
                    # Fallback: substring match
                    matches[label_str] = target_class.lower() in label_str.lower()

            return matches
        else:
            # Failed to parse JSON, fallback to substring matching
            print(f"[WARNING] Failed to parse JSON response for {target_class}, using fallback")
            return {str(label): target_class.lower() in str(label).lower() for label in instance_labels}

    except Exception as e:
        print(f"[ERROR] Failed to query LLaVA batch for {target_class}: {e}")
        # Fallback to substring matching
        return {str(label): target_class.lower() in str(label).lower() for label in instance_labels}


def load_instance_data(npz_dir: str, scene_id: str) -> Dict:
    """
    Load instance data from npz file

    Args:
        npz_dir: directory containing npz files
        scene_id: scene identifier

    Returns:
        dict with keys: ins_pcds, ins_labels, ins_scores, bbox8, etc.
        or None if file doesn't exist
    """
    npz_path = os.path.join(npz_dir, f"{scene_id}.npz")
    if not os.path.exists(npz_path):
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        return {
            'ins_pcds': data['ins_pcds'],
            'ins_labels': data['ins_labels'],
            'ins_scores': data['ins_scores'],
            'bbox8': data.get('bbox8', None),
            'views_per_instance': data.get('views_per_instance', None),
            'bbox2d_per_instance': data.get('bbox2d_per_instance', None),
            'ranging_ratio_per_instance': data.get('ranging_ratio_per_instance', None),
            'vis_ratio_per_instance': data.get('vis_ratio_per_instance', None),
        }
    except Exception as e:
        print(f"[ERROR] Failed to load {npz_path}: {e}")
        return None


def filter_instances_for_scene(
    model,
    tokenizer,
    conv_mode: str,
    target_class: str,
    instance_labels: list,
    use_exact_match: bool = False,
    temperature: float = 0.0,
    cache: Dict = None
) -> list:
    """
    Filter instances based on semantic matching with target class

    Args:
        model: LLaVA model
        tokenizer: tokenizer
        conv_mode: conversation mode
        target_class: target object class
        instance_labels: list of instance labels
        use_exact_match: if True, only use exact string matching (baseline)
        temperature: sampling temperature
        cache: optional dict to cache LLaVA results

    Returns:
        list of unique class names that match
    """
    if target_class == 'unknown' or not target_class:
        return []

    # Get unique labels
    unique_labels = list(set(str(label) for label in instance_labels))

    if use_exact_match:
        # Baseline: exact string matching
        matched_labels = []
        for label_str in unique_labels:
            if label_str.lower() == target_class.lower():
                matched_labels.append(label_str)
        return sorted(matched_labels)

    # Semantic matching with batching
    matched_labels = []

    # Check cache first and collect uncached labels
    uncached_labels = []

    for label_str in unique_labels:
        cache_key = (target_class.lower(), label_str.lower())

        if cache is not None and cache_key in cache:
            # Use cached result
            if cache[cache_key]:
                matched_labels.append(label_str)
        else:
            # Need to query
            uncached_labels.append(label_str)

    # Batch query for uncached labels
    if uncached_labels:
        match_results = query_llava_batch_semantic_match(
            model, tokenizer, conv_mode, target_class, uncached_labels, temperature
        )

        # Process results
        for label_str, is_match in match_results.items():
            # Cache result
            cache_key = (target_class.lower(), label_str.lower())
            if cache is not None:
                cache[cache_key] = is_match

            # Add to matched labels if YES
            if is_match:
                matched_labels.append(label_str)

    print(f"target class: {target_class}  |  matched_labels: {matched_labels}")
    return sorted(matched_labels)


def main(args):
    # Load dataset configuration from YAML (merges base.yaml + dataset yaml)
    cfg = load_dataset_config(args.dataset)
    llava_grounding_dir = cfg.get("llava_grounding_dir", "")

    # Extract paths from config
    # ScanRefer uses mask3d_pred_path, NR3D uses gt_instance_path
    if args.dataset == "scanrefer":
        npz_dir = args.npz_dir or cfg.get("mask3d_pred_path", "")
    else:
        npz_dir = args.npz_dir or cfg.get("gt_instance_path", "")

    # Construct full paths for parsed_file and output_file
    parsed_file = args.parsed_file or os.path.join(
        llava_grounding_dir, cfg.get("parsed_file", "")
    )
    output_file = args.output_file or os.path.join(
        llava_grounding_dir, cfg.get("parsed_with_candidate_file", "")
    )

    conv_mode = args.conv_mode or "qwen_1_5"
    model_path = args.model_path or MODEL_PATH
    dataset_name = cfg.get("dataset", args.dataset)

    # Validate input paths
    print(f"\n{'='*60}")
    print(f"Configuration ({dataset_name}):")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Parsed file: {parsed_file}")
    print(f"Instance NPZ dir: {npz_dir}")
    print(f"Output file: {output_file}")
    print(f"Conv mode: {conv_mode}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")

    if not os.path.exists(parsed_file):
        raise FileNotFoundError(f"Parsed file not found: {parsed_file}")

    if not os.path.exists(npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Load model
    print("Loading LLaVA model...")
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, attn_implementation="sdpa"
    )
    print(f"Model loaded: {model_name}\n")

    # Load parsed descriptions
    print(f"Loading parsed descriptions from {parsed_file}...")
    with open(parsed_file, 'r') as f:
        parsed_data = json.load(f)
    print(f"Loaded {len(parsed_data)} parsed descriptions")

    # Prepare output
    results = []

    # Cache for LLaVA semantic matching results
    semantic_match_cache = {}

    # Scene data cache
    scene_cache = {}

    # Statistics
    stats = {
        'total_samples': 0,
        'samples_with_instances': 0,
        'samples_with_instances_exact': 0,
        'total_instances_exact': 0,
        'total_instances_semantic': 0,
        'scenes_processed': set(),
        'improvement_by_class': defaultdict(lambda: {'exact': 0, 'semantic': 0}),
    }

    # Process each sample
    print(f"\nProcessing samples...")
    for sample_idx, sample in enumerate(tqdm(parsed_data, desc="Filtering instances")):
        if sample is None:
            results.append(None)
            continue

        scene_id = sample.get('scene_id')
        object_id = sample.get('object_id')
        ann_id = sample.get('ann_id')
        description = sample.get('description', '')
        prediction = sample.get('prediction', {})
        target_class = prediction.get('target_class', 'unknown')

        stats['total_samples'] += 1

        # Load instance data for this scene
        if scene_id not in stats['scenes_processed']:
            instance_data = load_instance_data(npz_dir, scene_id)
            scene_cache[scene_id] = instance_data
            stats['scenes_processed'].add(scene_id)
        else:
            instance_data = scene_cache.get(scene_id)

        if instance_data is None:
            print(f"[WARNING] No instance data found for scene {scene_id}")
            # Keep all original fields
            result = sample.copy() if sample else {}
            # Add empty candidate_class to prediction
            if 'prediction' in result and isinstance(result['prediction'], dict):
                result['prediction']['candidate_class'] = []
            else:
                result['prediction'] = {'candidate_class': []}
            results.append(result)
            continue

        instance_labels = instance_data['ins_labels']

        # Filter with exact matching (baseline)
        exact_matches = filter_instances_for_scene(
            model, tokenizer, conv_mode, target_class, instance_labels,
            use_exact_match=True, temperature=args.temperature
        )

        # Filter with semantic matching (using LLaVA)
        semantic_matches = filter_instances_for_scene(
            model, tokenizer, conv_mode, target_class, instance_labels,
            use_exact_match=False, temperature=args.temperature, cache=semantic_match_cache
        )

        # IMPORTANT: Ensure target_class is in candidate_class if it exists in instance_labels
        # This handles cases where LLaVA might miss exact matches
        if target_class and target_class != 'unknown':
            target_lower = target_class.lower()
            # Check if target_class exists in instance_labels (case-insensitive)
            target_in_labels = any(str(label).lower() == target_lower for label in instance_labels)
            # Check if target_class is already in semantic_matches (case-insensitive)
            target_in_matches = any(match.lower() == target_lower for match in semantic_matches)

            if target_in_labels and not target_in_matches:
                # Find the exact label string from instance_labels to maintain consistency
                for label in instance_labels:
                    if str(label).lower() == target_lower:
                        semantic_matches.append(str(label))
                        break
                semantic_matches = sorted(list(set(semantic_matches)))

        stats['total_instances_exact'] += len(exact_matches)
        stats['total_instances_semantic'] += len(semantic_matches)
        if len(exact_matches) > 0:
            stats['samples_with_instances_exact'] += 1
        if len(semantic_matches) > 0:
            stats['samples_with_instances'] += 1

        # Track improvement by target class
        if target_class and target_class != 'unknown':
            stats['improvement_by_class'][target_class]['exact'] += len(exact_matches)
            stats['improvement_by_class'][target_class]['semantic'] += len(semantic_matches)

        # Prepare result - keep all original fields
        result = sample.copy() if sample else {}

        # Add candidate_class to prediction field
        if 'prediction' in result and isinstance(result['prediction'], dict):
            result['prediction']['candidate_class'] = semantic_matches
        else:
            result['prediction'] = {'candidate_class': semantic_matches}

        results.append(result)

        # Periodic save
        if (sample_idx + 1) % args.save_interval == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n[SAVE] Saved {len(results)} results to {output_file}")

    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print statistics
    print(f"\n{'='*60}")
    print("Filtering Statistics:")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with candidates (exact): {stats['samples_with_instances_exact']}")
    print(f"Samples with candidates (semantic): {stats['samples_with_instances']}")
    print(f"Total candidate instances (exact match): {stats['total_instances_exact']}")
    print(f"Total candidate instances (semantic match): {stats['total_instances_semantic']}")
    improvement = stats['total_instances_semantic'] - stats['total_instances_exact']
    pct_increase = (stats['total_instances_semantic'] / max(stats['total_instances_exact'], 1) - 1) * 100
    print(f"Improvement: +{improvement} instances ({pct_increase:.1f}% increase)")
    print(f"Scenes processed: {len(stats['scenes_processed'])}")
    print(f"Cache size: {len(semantic_match_cache)} unique class-label pairs")
    print(f"{'='*60}")

    # Show per-class improvements
    print(f"\n{'='*60}")
    print("Top Classes by Improvement (Semantic - Exact):")
    print(f"{'='*60}")
    improvements = []
    for cls, counts in stats['improvement_by_class'].items():
        diff = counts['semantic'] - counts['exact']
        if diff > 0:
            improvements.append((cls, counts['exact'], counts['semantic'], diff))

    improvements.sort(key=lambda x: x[3], reverse=True)
    for cls, exact, semantic, diff in improvements[:15]:
        pct = diff / max(exact, 1) * 100
        print(f"  {cls:20s}: {exact:4d} -> {semantic:4d} (+{diff:3d}, {pct:5.1f}% increase)")

    if not improvements:
        print("  (No improvements found)")
    print(f"{'='*60}")

    # Show some examples of semantic matching improvements
    print(f"\n{'='*60}")
    print("Sample Semantic Matches (Non-Exact):")
    print(f"{'='*60}")
    count = 0
    shown = set()
    for key, value in semantic_match_cache.items():
        target, label = key
        if value and target != label and target not in shown:
            print(f"  '{target}' -> '{label}'")
            shown.add(target)
            count += 1
            if count >= 20:
                break

    if count == 0:
        print("  (No non-exact matches found in cache)")
    print(f"{'='*60}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct target category candidates using LLaVA for semantic matching"
    )

    # Dataset selection
    parser.add_argument("--dataset", type=str, default='scanrefer',
                        choices=["scanrefer", "nr3d"],
                        help="Dataset to process: 'scanrefer' or 'nr3d'")

    # Model configuration
    parser.add_argument("--model-path", type=str, default=None,
                        help=f"Path to LLaVA model (default: from config)")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (if applicable)")
    parser.add_argument("--conv-mode", type=str, default=None,
                        help="Conversation mode for model (default: from config)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for deterministic)")

    # Data paths (optional overrides)
    parser.add_argument("--parsed-file", type=str, default=None,
                        help="Path to parsed descriptions JSON (default: from config)")
    parser.add_argument("--npz-dir", type=str, default=None,
                        help="Directory containing instance NPZ files (default: from config)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output file path (default: from config)")

    # Runtime options
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save results every N samples")

    args = parser.parse_args()
    main(args)
