#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for 3D visual grounding: model path and prompts.
Dataset paths are loaded from YAML files (base.yaml + scanrefer.yaml/nr3d.yaml).
"""

import os
import yaml

# ============================================================
# Model Configuration
# ============================================================

MODEL_PATH = os.environ.get("QGROUND_MODEL_PATH", "/path/to/pretrained/llava-onevision-qwen2-7b-ov")

# ============================================================
# YAML Config Paths
# ============================================================

CONFIG_DIR = os.environ.get("QGROUND_CONFIG_DIR", os.path.dirname(__file__))
BASE_CONFIG = os.path.join(CONFIG_DIR, "base.yaml")
DATASET_CONFIGS = {
    "scanrefer": os.path.join(CONFIG_DIR, "scanrefer.yaml"),
    "nr3d": os.path.join(CONFIG_DIR, "nr3d.yaml"),
}


def load_yaml(yaml_path):
    """Load a single YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_base_config():
    """Load only the base configuration."""
    return load_yaml(BASE_CONFIG)


def load_dataset_config(dataset_type):
    """
    Load dataset configuration by merging base.yaml with dataset-specific yaml.

    Args:
        dataset_type: "scanrefer" or "nr3d"

    Returns:
        dict: Merged configuration with base settings and dataset-specific settings
    """
    # Load base config
    config = load_yaml(BASE_CONFIG)

    # Load and merge dataset-specific config
    dataset_yaml = DATASET_CONFIGS[dataset_type]
    dataset_config = load_yaml(dataset_yaml)
    config.update(dataset_config)

    return config


# ============================================================
# Prompts
# ============================================================

Description_parse_prompt = """You are working on a 3D visual grounding task, which involves receiving a query that specifies a particular object by describing its attributes and grounding conditions to uniquely identify the object. Here, attributes refer to the inherent properties of the object, such as category, color, appearance, function, etc. Grounding conditions refer to considerations of other objects or other conditions in the scene, such as location, relative position to other objects, etc.

Your tasks are:
1. Parse the query and extract the target object category, attributes, and grounding conditions. Each attribute and condition should be returned individually. Sometimes the object's category is not explicitly specified, and you need to deduce it through reasoning.
2. Generate a list of verification questions that can be used to verify if an object in an image matches this description

Each verification question should be:
- Clear and specific
- Answerable with YES/NO by examining the object in images
- Focused on one aspect (class, single attribute, or single condition)

Your response should be formatted using JSON contained in ```json```.

Here are some examples:

Input:
Query: this is a brown cabinet. it is to the right of a picture.

Output:
```
{
  "target_class": "cabinet",
  "attributes": ["it's brown"],
  "conditions": ["it's to the right of a picture"],
  "verification_questions": [
    "Is the object in the red bounding box a cabinet?",
    "Is the cabinet in the red bounding box brown?",
    "Is the cabinet in the red bounding box located to the right of a picture?"
  ]
}
```

Input:
Query: it is a wooden computer desk. the desk is in the sleeping area, across from the living room. the desk is in the corner of the room, between the nightstand and where the shelf and window are.

Output:
```
{
  "target_class": "desk",
  "attributes": ["it's a wooden computer desk"],
  "conditions": ["it's in the sleeping area", "it's across from the living room", "it's in the corner of the room", "it's between the nightstand and where the shelf and window are"],
  "verification_questions": [
    "Is the object in the red bounding box a desk?",
    "Is the desk in the red bounding box a wooden computer desk?",
    "Is the desk in the red bounding box located in the sleeping area?",
    "Is the desk in the red bounding box located across from the living room?",
    "Is the desk in the red bounding box located in the corner of the room?",
    "Is the desk in the red bounding box positioned between the nightstand and where the shelf and window are?"
  ]
}
```

Input:
Query: In the room is a set of desks along a wall with windows totaling 4 desks. Opposite this wall is another wall with a door and two desks. The desk of interest is the closest desk to the door. This desk has nothing on it, no monitor, etc.

Output:
```
{
  "target_class": "desk",
  "attributes": [],
  "conditions": ["it's near the wall with a door and two desks, opposite a wall with windows totaling four desks", "it's closest to the door on the wall", "it has nothing on it, no monitor, etc"],
  "verification_questions": [
    "Is the object in the red bounding box a desk?",
    "Is the desk in the red bounding box near the wall with a door and two desks, opposite a wall with windows totaling four desks?",
    "Is the desk in the red bounding box the one closest to the door on that wall?",
    "Does the desk in the red bounding box have nothing on it, no monitor, etc?"
  ]
}
```

Ensure your response adheres strictly to this JSON format, as it will be directly parsed and used. The verification_questions list should include exactly one question for the target_class, one question for each attribute, and one question for each condition.
"""




Category_match_prompt = """You are a strict semantic label matcher.

Rule:
- Return "YES" only if the instance label and target class refer to the SAME physical object type.
- Otherwise, return "NO".

Guidelines:
YES examples:
  chair ↔ chair
  shoe ↔ shoes
  cabinet ↔ kitchen cabinet

NO examples:
  desk ≠ chair
  chair ≠ table
  tv ≠ monitor

Output JSON: {"label": "YES" or "NO"} with no extra text."""


Verification_prompt = """You are an assistant for visual grounding and object verification. You will be shown multiple images of the SAME object from different viewpoints, with the object marked by a RED BOUNDING BOX in each image.

IMPORTANT: Focus ONLY on the object inside the red bounding box.

Your task is to answer verification questions about the marked object. Answer each question separately with YES or NO.

Instructions:
1. Carefully examine ONLY the object inside the red bounding box across all views
2. Answer EACH question separately with YES or NO
3. For spatial relationship questions, check if the relationships are visible in the images
4. If uncertain about any question, answer NO

Output format:
Q1: YES/NO
Q2: YES/NO
Q3: YES/NO
..."""

