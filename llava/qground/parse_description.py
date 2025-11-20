#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse 3D visual grounding descriptions using LLaVA model.
Supports both ScanRefer (JSON) and NR3D (CSV) datasets.

Output: Structured JSON with target_class, attributes, conditions, and verification_questions.
"""

import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg", force=True)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.dataset.config import MODEL_PATH, Description_parse_prompt, load_dataset_config
from typing import Dict
import transformers
import re


# ============================================================
# Validation
# ============================================================

def validate_parsed_output(parsed):
    """
    Validate parsed output to ensure it meets requirements.

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(parsed, dict):
        return False, "Parsed output is not a dictionary"

    # Check required fields
    required_fields = ['target_class', 'attributes', 'conditions']
    for field in required_fields:
        if field not in parsed:
            return False, f"Missing required field: {field}"

    # Validate target_class is a string, not a list
    if isinstance(parsed['target_class'], list):
        return False, f"target_class is a list: {parsed['target_class']}, should be a single string"

    if not isinstance(parsed['target_class'], str):
        return False, f"target_class is not a string: {type(parsed['target_class'])}"

    # Validate target_class is not empty
    if not parsed['target_class'] or parsed['target_class'].strip() == '':
        return False, "target_class is empty"

    # Validate target_class is not "unknown"
    if parsed['target_class'].lower().strip() == 'unknown':
        return False, "target_class is 'unknown' - needs re-parsing with better inference"

    # Validate attributes and conditions are lists
    if not isinstance(parsed['attributes'], list):
        return False, f"attributes is not a list: {type(parsed['attributes'])}"

    if not isinstance(parsed['conditions'], list):
        return False, f"conditions is not a list: {type(parsed['conditions'])}"

    # Validate verification_questions
    if 'verification_questions' in parsed:
        if not isinstance(parsed['verification_questions'], list):
            return False, f"verification_questions is not a list: {type(parsed['verification_questions'])}"

    return True, None


# ============================================================
# Dataset Loading
# ============================================================

def load_scanrefer_dataset(json_path):
    """Load ScanRefer dataset from JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    dataset_info = []
    for item in data:
        dataset_info.append({
            'scene_id': item['scene_id'],
            'object_id': str(item['object_id']),
            'ann_id': str(item['ann_id']),
            'description': item['description'],
            'instance_type': item.get('instance_type', ''),
        })
    return dataset_info


def load_nr3d_dataset(csv_path):
    """Load NR3D dataset from CSV file."""
    df = pd.read_csv(csv_path)

    dataset_info = []
    for idx, row in df.iterrows():
        item = {
            'scene_id': row['scan_id'],
            'object_id': str(row['target_id']),
            'ann_id': str(row.get('ann_id', idx)),
            'description': row['utterance'],
            'instance_type': row.get('instance_type', ''),
        }
        # NR3D-specific fields
        if 'is_easy' in row:
            item['is_easy'] = row['is_easy']
        if 'is_view_dep' in row:
            item['is_view_dep'] = row['is_view_dep']
        # SR3D-specific fields
        if 'reference_type' in row:
            item['reference_type'] = row['reference_type']
        if 'coarse_reference_type' in row:
            item['coarse_reference_type'] = row['coarse_reference_type']
        if 'distractor_ids' in row:
            item['distractor_ids'] = row['distractor_ids']
        if 'anchor_ids' in row:
            item['anchor_ids'] = row['anchor_ids']

        dataset_info.append(item)

    return dataset_info


# ============================================================
# Tokenization
# ============================================================

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer,
                    has_image: bool = False, system_message: str = "You are a helpful assistant.") -> Dict:
    """Preprocess input for Qwen-based LLaVA model."""
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

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


# ============================================================
# Main Evaluation
# ============================================================

def eval_model(args):
    dataset_type = args.dataset_type
    split = args.split

    # Load dataset config from YAML
    cfg = load_dataset_config(dataset_type)

    print(f"Dataset type: {dataset_type}")
    print(f"Split: {split}")

    # Determine input file path from YAML config
    data_dir = cfg['data_dir']
    data_files = cfg.get('data_files', {})
    if dataset_type == 'scanrefer':
        input_file = os.path.join(data_dir, data_files.get(split, f"ScanRefer_filtered_{split}.json"))
        dataset_info = load_scanrefer_dataset(input_file)
    else:
        input_file = os.path.join(data_dir, data_files.get(split, f"nr3d_{split}.csv"))
        dataset_info = load_nr3d_dataset(input_file)

    print(f"Input file: {input_file}")
    print(f"Loaded {len(dataset_info)} samples")

    # Determine save file path from YAML config or argument
    if args.output_file:
        save_file = args.output_file
        output_dir = os.path.dirname(save_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = cfg['llava_grounding_dir']
        os.makedirs(output_dir, exist_ok=True)
        save_file = os.path.join(output_dir, f"description_parsing_{split}.json")

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(MODEL_PATH)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    print(f"Save path: {save_file}")

    # Load existing results (for resume)
    per_item_results = []

    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            per_item_results = json.load(f)
        print(f"Loaded {len(per_item_results)} existing results from {save_file}")

    # Determine samples to process (continue from where we left off)
    samples_to_process = list(range(len(per_item_results), len(dataset_info)))
    if not samples_to_process:
        print("All samples already processed!")
        return

    print(f"Processing {len(samples_to_process)} remaining samples...")

    # Process samples
    for idx in tqdm(samples_to_process, desc="Processing descriptions"):
        item = dataset_info[idx]
        scene_id = item['scene_id']
        object_id = item['object_id']
        description = item['description']
        ann_id = item['ann_id']

        # Retry logic
        max_retries = 5
        parsed = None
        outputs = None

        for retry in range(max_retries):
            try:
                conv = conv_templates[args.conv_mode].copy()
                line = {'from': 'human', 'value': f'Description: {description}'}

                input_ids = preprocess_qwen(
                    [line, {'from': 'gpt', 'value': None}],
                    tokenizer,
                    has_image=False,
                    system_message=Description_parse_prompt
                ).cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=None,
                        do_sample=True,
                        temperature=0.2,
                        num_beams=1,
                        max_new_tokens=1024,
                        use_cache=True
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]

                # Extract JSON
                json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", outputs)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = re.search(r"\{[\s\S]*?\}", outputs).group(0)

                json_clean = re.sub(r'#.*', '', json_str)
                parsed = json.loads(json_clean)

                # Validate
                is_valid, error_msg = validate_parsed_output(parsed)
                if is_valid:
                    break
                else:
                    print(f"\n[Retry {retry+1}/{max_retries}] Validation failed for {scene_id}: {error_msg}")
                    if retry < max_retries - 1:
                        if isinstance(parsed.get('target_class'), list) and len(parsed['target_class']) > 0:
                            parsed['target_class'] = parsed['target_class'][0]
                            print(f"  Auto-fix: Using first class from list: {parsed['target_class']}")
                            is_valid, _ = validate_parsed_output(parsed)
                            if is_valid:
                                break
                    parsed = None

            except Exception as e:
                print(f"\n[Error {retry+1}/{max_retries}] Failed to parse {scene_id}: {str(e)}")
                if retry == max_retries - 1:
                    parsed = {
                        'target_class': 'unknown',
                        'attributes': [],
                        'conditions': [],
                        'verification_questions': ["Is the object in the red bounding box identifiable?"],
                        'parse_error': str(e)
                    }

        # Build result
        result = {
            "scene_id": scene_id,
            "object_id": object_id,
            "ann_id": ann_id,
            "description": description,
            "raw_prediction": outputs if outputs else "",
            "prediction": parsed if parsed else {}
        }

        # Add instance_type only for NR3D (not ScanRefer)
        if dataset_type == 'nr3d':
            result["instance_type"] = item.get('instance_type', '')

        # Add dataset-specific fields
        for field in ['is_easy', 'is_view_dep', 'reference_type', 'coarse_reference_type',
                      'distractor_ids', 'anchor_ids']:
            if field in item:
                result[field] = item[field]

        # Ensure list has correct length
        while len(per_item_results) <= idx:
            per_item_results.append(None)
        per_item_results[idx] = result

        # Save periodically
        if (len(samples_to_process) > 100 and (samples_to_process.index(idx) + 1) % 10 == 0) or \
           (len(samples_to_process) <= 100 and (samples_to_process.index(idx) + 1) % 5 == 0):
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(per_item_results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(per_item_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(per_item_results)} results to {save_file}")

    # Final validation report
    print(f"\n{'='*60}")
    print("Final Validation Report:")
    print(f"{'='*60}")
    invalid_final = 0
    none_count = 0
    for result in per_item_results:
        if result is None:
            none_count += 1
        elif 'prediction' in result and result['prediction']:
            is_valid, _ = validate_parsed_output(result['prediction'])
            if not is_valid:
                invalid_final += 1

    print(f"Total samples: {len(per_item_results)}")
    print(f"Valid samples: {len(per_item_results) - invalid_final - none_count}")
    print(f"Invalid samples: {invalid_final}")
    print(f"None/unprocessed samples: {none_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse 3D visual grounding descriptions (ScanRefer/NR3D)"
    )
    parser.add_argument("--dataset-type", type=str, default="scanrefer",
                        choices=["scanrefer", "nr3d"],
                        help="Dataset type (default: scanrefer)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split (default: val)")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                        help="Conversation mode")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Override output file path (optional)")

    args = parser.parse_args()
    eval_model(args)
