#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QGround instance verification with 4D masked attention scoring.

For each candidate instance (filtered by `candidate_class` from G1.5), we run a
single forward pass over its top-K image views with a custom 4D attention mask
that isolates each verification question. YES/NO probabilities are read
directly from logits at each question's "Answer:" position — no generation.

Per-candidate score = sum_i (yes_prob_i - no_prob_i) ("model preference").
The candidate with the highest preference wins; its 3D AABB is the prediction.

Dataset-aware Q1 skip:
  - ScanRefer: ask all questions, including the category question (Q1).
  - Nr3D:      skip Q1 by default. Q1 stays in the parsed file but is not
               asked at inference, matching the published Skip-Q1 setting.

Requires `attn_implementation="sdpa"` (FlashAttention-2 does not support 4D
attention masks).

Usage:
    python instances_verification.py --dataset scanrefer
    python instances_verification.py --dataset nr3d
"""

import argparse
import os
import re
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.dataset.config import MODEL_PATH, load_dataset_config


# =============================================================================
# Image utilities
# =============================================================================

def draw_bbox_on_image(image_path, bbox, line_width=5):
    """Draw a red bounding box on the image."""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    x_min, y_min = bbox['x_min'], bbox['y_min']
    x_max, y_max = bbox['x_max'], bbox['y_max']
    for i in range(line_width):
        draw.rectangle([x_min - i, y_min - i, x_max + i, y_max + i],
                       outline='red', width=1)
    return image


# =============================================================================
# YES/NO token IDs (multiple variants summed)
# =============================================================================

def get_yes_no_token_ids(tokenizer):
    """Pre-compute first-token IDs for YES/NO answer variants."""
    yes_ids, no_ids = set(), set()
    for text in ["YES", "Yes", "yes", " YES", " Yes", " yes", "Y", " Y"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            yes_ids.add(ids[0])
    for text in ["NO", "No", "no", " NO", " No", " no", "N", " N"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            no_ids.add(ids[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids -= overlap
    return list(yes_ids), list(no_ids)


# =============================================================================
# Build input_ids with per-question position tracking
# =============================================================================

def build_input_ids_with_positions(tokenizer, system_msg, num_images,
                                   verification_questions):
    """
    Tokenize a multi-image, multi-question prompt and return:
      input_ids                 - (1, S) full sequence
      question_boundaries_in_tail - list of (start, end) offsets in `tail_tokens`
      intro_token_count         - intro length (the "\n\n" between images and Qs)
      tail_len                  - len(tail_tokens) = intro + per-Q text + per-Q
                                  turn-suffix tokens
      nl_after_images_len       - newlines that follow the last image in the
                                  expanded multimodal sequence

    Each question block tokens = "Q{i}: <text>\nAnswer:" + turn_suffix where
    turn_suffix = <|im_end|>\n<|im_start|>assistant\n  (so each question sees
    a complete user→assistant turn closure).
    """
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids

    turn_suffix_tokens = (
        [im_end] + nl_tokens
        + tokenizer("<|im_start|>assistant").input_ids + nl_tokens
    )

    system_tokens = (
        [im_start] + tokenizer("system").input_ids + nl_tokens
        + tokenizer(system_msg).input_ids + [im_end] + nl_tokens
    )
    user_header = tokenizer("<|im_start|>user").input_ids + nl_tokens

    empty_tokens = tokenizer("").input_ids
    image_section = []
    for _ in range(num_images):
        image_section += empty_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens

    intro_text = "\n\n"
    tail_text = intro_text
    intro_token_count = len(tokenizer(tail_text).input_ids)

    question_block_texts = [f"Q{i+1}: {q}\nAnswer:"
                            for i, q in enumerate(verification_questions)]

    text_boundaries_in_text = []
    for i, q_block in enumerate(question_block_texts):
        if i > 0:
            tail_text += "\n"
        start_count = len(tokenizer(tail_text).input_ids)
        tail_text += q_block
        end_count = len(tokenizer(tail_text).input_ids)
        text_boundaries_in_text.append((start_count, end_count))

    tail_text_tokens = tokenizer(tail_text).input_ids
    tail_tokens = list(tail_text_tokens)

    offset = 0
    question_boundaries_in_tail = []
    for ts, te in text_boundaries_in_text:
        adjusted_start = ts + offset
        adjusted_end = te + offset
        tail_tokens[adjusted_end:adjusted_end] = turn_suffix_tokens
        question_boundaries_in_tail.append(
            (adjusted_start, adjusted_end + len(turn_suffix_tokens))
        )
        offset += len(turn_suffix_tokens)

    tail_len = len(tail_tokens)
    nl_after_images_len = len(nl_tokens)

    full_tokens = system_tokens + user_header + image_section + tail_tokens
    input_ids = torch.tensor([full_tokens], dtype=torch.long)

    return (input_ids, question_boundaries_in_tail,
            intro_token_count, tail_len, nl_after_images_len)


# =============================================================================
# Build 4D attention mask + custom position_ids
# =============================================================================

def build_4d_mask_and_position_ids(expanded_len, question_boundaries,
                                   intro_token_count, tail_len,
                                   nl_after_images_len, dtype, device):
    """
    Construct a 4D attention mask and (1, S) position_ids such that:
      - Shared prefix [0, P): standard causal attention.
      - Each Qi block: causal within itself; attends to the entire shared
                       prefix; CANNOT attend to other Q blocks.
      - Position IDs: shared prefix [0..P-1]; each Qi restarts from P, so
                       every question has identical RoPE distance to the
                       visual tokens.

    The mask is returned as a (1, 1, S, S) BOOL tensor (True = attend,
    False = block). LLaMA-style SdpaAttention in transformers reliably
    respects bool 4D masks; the float-additive form goes through a different
    code path that can be overridden by `is_causal=True` and silently fall
    back to plain causal attention. `dtype` is kept in the signature for API
    compatibility but is no longer used for the mask itself.
    """
    S = expanded_len
    num_questions = len(question_boundaries)

    full_tail_len = nl_after_images_len + tail_len
    tail_start = S - full_tail_len
    tail_tokens_start = tail_start + nl_after_images_len
    shared_prefix_end = tail_tokens_start + intro_token_count  # P

    q_ranges = []
    for (qs_rel, qe_rel) in question_boundaries:
        q_ranges.append((tail_tokens_start + qs_rel,
                         tail_tokens_start + qe_rel))

    mask_4d = torch.zeros((1, 1, S, S), dtype=torch.bool, device=device)
    P = shared_prefix_end

    # Causal within shared prefix
    prefix_idx = torch.arange(P, device=device)
    mask_4d[0, 0, :P, :P] = prefix_idx.unsqueeze(0) <= prefix_idx.unsqueeze(1)

    # All non-prefix tokens see the full shared prefix
    mask_4d[0, 0, P:, :P] = True

    # Causal within each question block
    for (qs, qe) in q_ranges:
        bl = qe - qs
        if bl <= 0:
            continue
        block_idx = torch.arange(bl, device=device)
        mask_4d[0, 0, qs:qe, qs:qe] = block_idx.unsqueeze(0) <= block_idx.unsqueeze(1)

    # Position IDs
    position_ids = torch.zeros((1, S), dtype=torch.long, device=device)
    position_ids[0, :P] = torch.arange(P, device=device)
    for (qs, qe) in q_ranges:
        bl = qe - qs
        if bl <= 0:
            continue
        position_ids[0, qs:qe] = torch.arange(P, P + bl, device=device)
    # Separator tokens between questions: assigned position P (their outputs
    # are not used).
    for i in range(num_questions - 1):
        sep_start = q_ranges[i][1]
        sep_end = q_ranges[i + 1][0]
        if sep_end > sep_start:
            position_ids[0, sep_start:sep_end] = P

    return mask_4d, position_ids, q_ranges


# =============================================================================
# Per-instance verification (single forward pass)
# =============================================================================

VERIFY_SYSTEM_MSG = (
    "You are an assistant for visual grounding and object verification. "
    "You will be shown multiple images of the SAME object from different "
    "viewpoints, with the object marked by a RED BOUNDING BOX in each image.\n\n"
    "IMPORTANT: Focus ONLY on the object inside the red bounding box.\n\n"
    "Your task is to answer a verification question about the marked object. "
    "Answer with YES or NO only.\n\n"
    "Instructions:\n"
    "1. Carefully examine ONLY the object inside the red bounding box across all views\n"
    "2. Answer the question with YES or NO\n"
    "3. For spatial relationship questions, check if the relationships are visible in the images\n"
    "4. If uncertain, answer NO"
)


def verify_instance_with_views_mask_attention(
    model, tokenizer, image_processor,
    parsed_description, instance_views, scene_id, image_root,
    top_k, yes_token_ids, no_token_ids, skip_first_question,
):
    """
    Verify if a single instance matches the parsed description by examining its
    top-K views with a single 4D-masked-attention forward pass.

    Returns a dict with: view_info, final_decision, raw_output, individual_answers,
    individual_scores, questions, num_questions, num_valid_views, method.
    """
    selected_views = instance_views[:top_k]
    view_info_list = []
    images_with_bbox = []

    for v in selected_views:
        view_id = v['view_id']
        bbox_2d = v['bbox_2d']
        vis_ratio = v['vis_ratio']

        if bbox_2d is None:
            view_info_list.append({'view_id': view_id, 'vis_ratio': vis_ratio,
                                   'skipped': True, 'reason': 'NO_BBOX'})
            continue

        image_path = os.path.join(image_root, scene_id, 'color', f'{view_id}.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(image_root, scene_id, 'color', f'{view_id}.png')
            if not os.path.exists(image_path):
                view_info_list.append({'view_id': view_id, 'vis_ratio': vis_ratio,
                                       'skipped': True, 'reason': 'IMAGE_NOT_FOUND'})
                continue

        try:
            img = draw_bbox_on_image(image_path, bbox_2d)
            images_with_bbox.append(img)
            view_info_list.append({'view_id': view_id, 'vis_ratio': vis_ratio,
                                   'skipped': False})
        except Exception as e:
            print(f"[ERROR] view {view_id}: {e}")
            view_info_list.append({'view_id': view_id, 'vis_ratio': vis_ratio,
                                   'skipped': True, 'reason': 'LOAD_ERROR',
                                   'error': str(e)})

    if not images_with_bbox:
        return {'view_info': view_info_list, 'final_decision': 'NO_VALID_VIEWS',
                'raw_output': None, 'num_valid_views': 0,
                'method': 'mask_attention_4d'}

    try:
        # ---------- assemble verification questions ----------
        if 'verification_questions' in parsed_description and parsed_description['verification_questions']:
            verification_questions = list(parsed_description['verification_questions'])
        else:
            target_class = parsed_description.get('target_class', 'object')
            attributes = parsed_description.get('attributes', [])
            conditions = parsed_description.get('conditions', [])
            verification_questions = [f"Is the object in the red bounding box a {target_class}?"]
            for a in attributes:
                verification_questions.append(f"Does the object have the following attribute: {a}?")
            for c in conditions:
                verification_questions.append(f"Does the object satisfy this condition: {c}?")

        # Per-dataset Q1 skip: drop the category question entirely (do not ask).
        if skip_first_question and len(verification_questions) >= 1:
            verification_questions = verification_questions[1:]

        num_questions = len(verification_questions)
        if num_questions == 0:
            return {'view_info': view_info_list, 'final_decision': 'NO_QUESTIONS',
                    'raw_output': None, 'num_valid_views': len(images_with_bbox),
                    'method': 'mask_attention_4d'}

        # ---------- build input_ids and 4D mask ----------
        num_images = len(images_with_bbox)
        (input_ids, question_boundaries, intro_token_count,
         tail_len, nl_after_images_len) = build_input_ids_with_positions(
            tokenizer, VERIFY_SYSTEM_MSG, num_images, verification_questions
        )

        # Process images
        vision_tower = model.get_vision_tower()
        vision_on_cpu = not next(vision_tower.parameters()).is_cuda
        image_tensors = []
        for img in images_with_bbox:
            t = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
            image_tensors.append(t.float() if vision_on_cpu else t.half())

        # Embed
        embed_device = model.get_model().embed_tokens.weight.device
        input_ids = input_ids.to(embed_device)
        torch.cuda.empty_cache()
        with torch.inference_mode():
            (_, _, _, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, image_tensors, ["image"]
            )

        expanded_len = inputs_embeds.shape[1]
        mask_4d, position_ids, q_ranges = build_4d_mask_and_position_ids(
            expanded_len=expanded_len,
            question_boundaries=question_boundaries,
            intro_token_count=intro_token_count, tail_len=tail_len,
            nl_after_images_len=nl_after_images_len,
            dtype=inputs_embeds.dtype, device=inputs_embeds.device,
        )

        # Single forward pass
        with torch.inference_mode():
            outputs = model(
                input_ids=None, inputs_embeds=inputs_embeds,
                attention_mask=mask_4d, position_ids=position_ids,
                past_key_values=None, labels=None, use_cache=False,
                output_attentions=False, output_hidden_states=False,
                images=None, return_dict=True,
            )
        logits = outputs.logits  # (1, S, vocab)

        # Extract YES/NO probabilities at each question's answer position
        individual_answers, individual_scores = [], []
        for qi in range(num_questions):
            qs, qe = q_ranges[qi]
            answer_pos = qe - 1  # last token of question block
            probs = torch.softmax(logits[0, answer_pos, :].float(), dim=-1)
            yes_prob = sum(probs[t].item() for t in yes_token_ids)
            no_prob = sum(probs[t].item() for t in no_token_ids)
            individual_answers.append('YES' if yes_prob > no_prob else 'NO')
            individual_scores.append({'yes_prob': round(yes_prob, 6),
                                      'no_prob': round(no_prob, 6)})

        if all(a == 'YES' for a in individual_answers):
            final_decision = 'YES'
        elif any(a == 'NO' for a in individual_answers):
            final_decision = 'NO'
        else:
            final_decision = 'UNCERTAIN'

        questions_with_prefix = [f"Q{i+1}: {q}"
                                 for i, q in enumerate(verification_questions)]
        combined_raw = '\n'.join(
            f"{qp} -> {ans} (yes={sc['yes_prob']:.4f}, no={sc['no_prob']:.4f})"
            for qp, ans, sc in zip(questions_with_prefix,
                                   individual_answers, individual_scores)
        )

        return {
            'view_info': view_info_list,
            'final_decision': final_decision,
            'raw_output': combined_raw,
            'individual_answers': individual_answers,
            'individual_scores': individual_scores,
            'questions': questions_with_prefix,
            'num_questions': num_questions,
            'num_valid_views': len(images_with_bbox),
            'method': 'mask_attention_4d',
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] mask-attention forward failed: {e}")
        traceback.print_exc()
        return {'view_info': view_info_list, 'final_decision': 'ERROR',
                'raw_output': None, 'error': str(e),
                'num_valid_views': len(images_with_bbox),
                'method': 'mask_attention_4d'}


# =============================================================================
# Load candidate instances from NPZ
# =============================================================================

VIEW_SCORE_MODES = {
    'combined':    lambda v: (float(v['vis_ratio']) + float(v['ranging_ratio'])) / 2.0,
    'vis_only':    lambda v: float(v['vis_ratio']),
    'ranging_only':lambda v: float(v['ranging_ratio']),
}


def load_instances_from_npz_with_candidates(npz_path, candidate_classes,
                                            top_k_views=10,
                                            view_score_mode='vis_only'):
    """Load candidate instances and rank their views by the chosen score mode."""
    data = np.load(npz_path, allow_pickle=True)
    cand_norm = [c.replace('_', ' ').lower().strip() for c in candidate_classes]
    score_fn = VIEW_SCORE_MODES[view_score_mode]
    instances = {}

    for inst_id, sem_label in enumerate(data['ins_labels']):
        if sem_label.replace('_', ' ').lower().strip() not in cand_norm:
            continue

        views_list = data['views_per_instance'][inst_id]
        bbox2d_dict = data['bbox2d_per_instance'][inst_id]
        vis_dict = data['vis_ratio_per_instance'][inst_id]
        rng_dict = data['ranging_ratio_per_instance'][inst_id]

        vinfos = []
        for view_id in views_list:
            if view_id not in bbox2d_dict or view_id not in vis_dict or view_id not in rng_dict:
                continue
            bb = bbox2d_dict[view_id]
            vinfos.append({
                'view_id': view_id,
                'vis_ratio': float(vis_dict[view_id]),
                'ranging_ratio': float(rng_dict[view_id]),
                'bbox_2d': {'x_min': float(bb[0]), 'y_min': float(bb[1]),
                            'x_max': float(bb[2]), 'y_max': float(bb[3])},
                'semantic_label': sem_label,
            })
        for v in vinfos:
            v['_score'] = score_fn(v)
        vinfos.sort(key=lambda v: v['_score'], reverse=True)
        vinfos = vinfos[:top_k_views]
        for v in vinfos:
            v.pop('_score', None)
        if vinfos:
            instances[str(inst_id)] = vinfos

    return instances


# =============================================================================
# Main
# =============================================================================

def main(args):
    cfg = load_dataset_config(args.dataset)
    llava_grounding_dir = cfg.get('llava_grounding_dir', '')
    color_root = cfg.get('color_root', '')
    npz_dir = (cfg.get('mask3d_pred_path', '') if args.dataset == 'scanrefer'
               else cfg.get('gt_instance_path', ''))

    parsed_file = os.path.join(llava_grounding_dir, cfg.get('parsed_with_candidate_file', ''))
    output_file = os.path.join(llava_grounding_dir, cfg.get('verification_file', ''))

    if args.parsed_file:  parsed_file = args.parsed_file
    if args.output_file:  output_file = args.output_file
    if args.npz_dir:      npz_dir = args.npz_dir
    if args.image_root:   color_root = args.image_root

    model_path = os.path.expanduser(args.model_path or MODEL_PATH)

    # Decide Q1-skip behavior. CLI overrides dataset default.
    if args.skip_category_question is None:
        skip_q1 = (args.dataset == 'nr3d')
    else:
        skip_q1 = bool(args.skip_category_question)

    print(f"\n{'='*70}")
    print(f"Configuration ({args.dataset}):")
    print(f"  model:          {model_path}")
    print(f"  parsed file:    {parsed_file}")
    print(f"  npz dir:        {npz_dir}")
    print(f"  image root:     {color_root}")
    print(f"  output file:    {output_file}")
    print(f"  top-K views:    {args.top_k}")
    print(f"  pool views:     {args.top_k_views_per_instance}")
    print(f"  view-score:     {args.view_score_mode}")
    print(f"  skip Q1 (cat):  {skip_q1}  (default for nr3d=True, scanrefer=False)")
    print(f"{'='*70}\n")

    if not os.path.exists(parsed_file):
        raise FileNotFoundError(f"Parsed file not found: {parsed_file}")
    if not os.path.exists(npz_dir):
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # ---- Load model (sdpa is required for 4D attention masks) ----
    print("Loading LLaVA model with attn_implementation=sdpa ...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        attn_implementation='sdpa',
    )
    print(f"Loaded {model_name} (context={context_len})")

    yes_token_ids, no_token_ids = get_yes_no_token_ids(tokenizer)
    print(f"YES token IDs: {yes_token_ids}")
    print(f"NO  token IDs: {no_token_ids}")

    # ---- Load parsed descriptions ----
    with open(parsed_file, 'r') as f:
        parsed_descriptions = json.load(f)
    if parsed_descriptions and 'object_id' not in parsed_descriptions[0]:
        raise ValueError("parsed_description_file must contain 'object_id' field.")
    print(f"Loaded {len(parsed_descriptions)} parsed descriptions")

    # Optional chunking for parallel runs
    if args.num_chunks > 1:
        parsed_descriptions = [it for i, it in enumerate(parsed_descriptions)
                               if i % args.num_chunks == args.chunk_idx]
        if args.chunk_idx >= 0:
            base = output_file[:-5] if output_file.endswith('.json') else output_file
            output_file = f"{base}_chunk{args.chunk_idx}_of_{args.num_chunks}.json"
        print(f"Chunk {args.chunk_idx}/{args.num_chunks}: {len(parsed_descriptions)} samples")
        print(f"Chunked output -> {output_file}")

    # ---- Resume support ----
    verification_results, prediction_results = [], {}
    processed_keys = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            verification_results = json.load(f)
        for r in verification_results:
            processed_keys.add((r['scene_id'], r['object_id'], r['ann_id']))
        print(f"Resuming from {len(verification_results)} existing results")

    # ---- Main loop ----
    for idx, item in enumerate(tqdm(parsed_descriptions,
                                    desc=f"Verifying {args.dataset} (mask_attention_4d)")):
        scene_id, object_id, ann_id = item['scene_id'], item['object_id'], item['ann_id']
        description = item['description']
        parsed_desc = item['prediction']
        sample_key = (scene_id, object_id, ann_id)
        if sample_key in processed_keys:
            continue

        target_class = parsed_desc.get('target_class', 'object')
        candidate_classes = parsed_desc.get('candidate_class') or [target_class]

        npz_path = os.path.join(npz_dir, f"{scene_id}.npz")
        if not os.path.exists(npz_path):
            verification_results.append({
                'scene_id': scene_id, 'object_id': object_id,
                'description': description, 'ann_id': ann_id,
                'num_instances': 0, 'instance_verification': {},
                'error': 'NPZ_FILE_NOT_FOUND',
            })
            continue

        try:
            instances = load_instances_from_npz_with_candidates(
                npz_path=npz_path, candidate_classes=candidate_classes,
                top_k_views=args.top_k_views_per_instance,
                view_score_mode=args.view_score_mode,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            verification_results.append({
                'scene_id': scene_id, 'object_id': object_id,
                'description': description, 'ann_id': ann_id,
                'num_instances': 0, 'instance_verification': {},
                'error': f'LOAD_ERROR: {e}',
            })
            continue

        if not instances:
            verification_results.append({
                'scene_id': scene_id, 'object_id': object_id,
                'description': description, 'ann_id': ann_id,
                'num_instances': 0, 'instance_verification': {},
                'note': 'NO_MATCHING_INSTANCES',
            })
            continue

        instance_verification = {}
        for inst_idx, instance_views in instances.items():
            instance_verification[inst_idx] = verify_instance_with_views_mask_attention(
                model=model, tokenizer=tokenizer, image_processor=image_processor,
                parsed_description=parsed_desc, instance_views=instance_views,
                scene_id=scene_id, image_root=color_root, top_k=args.top_k,
                yes_token_ids=yes_token_ids, no_token_ids=no_token_ids,
                skip_first_question=skip_q1,
            )

        verification_results.append({
            'scene_id': scene_id, 'object_id': object_id,
            'description': description, 'ann_id': ann_id,
            'num_instances': len(instances),
            'instance_verification': instance_verification,
        })

        # ---- Read-preference candidate selection ----
        # score(c) = sum_q (yes_prob_q - no_prob_q); argmax single best.
        instance_scores = {}
        for inst_idx, r in instance_verification.items():
            scs = r.get('individual_scores')
            if scs:
                instance_scores[inst_idx] = sum(s['yes_prob'] - s['no_prob']
                                                for s in scs)
            elif r.get('final_decision') == 'YES':
                instance_scores[inst_idx] = 1.0
            else:
                instance_scores[inst_idx] = -1.0

        if instance_scores:
            best_inst = max(instance_scores, key=instance_scores.get)
            try:
                bbox8_all = np.load(npz_path, allow_pickle=True)['bbox8']
                ii = int(best_inst)
                if ii < len(bbox8_all):
                    bb = bbox8_all[ii]
                    aabb = np.array([[bb.min(axis=0), bb.max(axis=0)]],
                                    dtype=np.float32)
                    prediction_results[(scene_id, object_id, ann_id)] = {'aabb_bound': aabb}
            except Exception as e:
                print(f"[WARNING] bbox load failed for {sample_key}: {e}")

        if (idx + 1) % args.save_interval == 0:
            with open(output_file, 'w') as f:
                json.dump(verification_results, f, indent=2)
            print(f"\n[SAVE] {len(verification_results)} -> {output_file}")

    with open(output_file, 'w') as f:
        json.dump(verification_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"[DONE] verification details: {len(verification_results)} -> {output_file}")

    if prediction_results:
        pred_path = output_file[:-5] + '_predictions.pkl' \
            if output_file.endswith('.json') else output_file + '_predictions.pkl'
        with open(pred_path, 'wb') as f:
            pickle.dump(prediction_results, f)
        print(f"[DONE] predictions: {len(prediction_results)} -> {pred_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="QGround instance verification (4D-masked-attention scoring)"
    )
    parser.add_argument('--dataset', type=str, default='scanrefer',
                        choices=['scanrefer', 'nr3d'])

    parser.add_argument('--model-path', type=str, default=None,
                        help="Override MODEL_PATH from llava/dataset/config.py")
    parser.add_argument('--model-base', type=str, default=None)

    parser.add_argument('--parsed-file', type=str, default=None,
                        help="Parsed-with-candidates JSON (overrides config)")
    parser.add_argument('--npz-dir', type=str, default=None)
    parser.add_argument('--image-root', type=str, default=None)
    parser.add_argument('--output-file', type=str, default=None)

    parser.add_argument('--top-k', type=int, default=3,
                        help="Top-K views per candidate (default 3)")
    parser.add_argument('--top-k-views-per-instance', type=int, default=10,
                        help="Pool size before top-K (default 10)")
    parser.add_argument('--view-score-mode', type=str, default='vis_only',
                        choices=list(VIEW_SCORE_MODES.keys()),
                        help="View ranking score: vis_only (default) | combined (vis+ranging)/2 | ranging_only")

    # Q1 skip — defaults to ON for nr3d, OFF for scanrefer.
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--skip-category-question', dest='skip_category_question',
                   action='store_true',
                   help="Force skip Q1 (the category question). Default: True for nr3d, False for scanrefer.")
    g.add_argument('--no-skip-category-question', dest='skip_category_question',
                   action='store_false',
                   help="Force ask Q1.")
    parser.set_defaults(skip_category_question=None)

    parser.add_argument('--save-interval', type=int, default=10)

    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)

    args = parser.parse_args()
    if args.chunk_idx < 0 or args.chunk_idx >= args.num_chunks:
        raise ValueError(f"chunk-idx ({args.chunk_idx}) must be in [0, {args.num_chunks})")
    main(args)
