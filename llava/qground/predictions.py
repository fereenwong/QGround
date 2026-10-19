"""Reconstruct single-box predictions from saved condition probabilities.

This function is shared by inference, resume and offline replay. It uses raw
YES/NO token probability mass (no additional binary renormalization).
"""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np


def candidate_scores(verification):
    """Score valid candidates; missing views or failed forwards cannot win."""
    result = {}
    for candidate, record in verification.items():
        scores = record.get('individual_scores')
        if record.get('error'):
            continue
        if not scores:
            # Released Nr3D Skip-Q1 artifacts retain a hard decision on four
            # records whose sole question was removed in post-processing.
            if record.get('final_decision') in ('YES', 'NO'):
                result[candidate] = 1.0 if record['final_decision'] == 'YES' else -1.0
            continue
        values = [float(s['yes_prob']) - float(s['no_prob']) for s in scores]
        if all(np.isfinite(values)):
            result[candidate] = sum(values)
    return result


def predictions_from_details(records, npz_dir):
    """Replay all records, preserving historical first-candidate tie breaking."""
    predictions, boxes = {}, {}
    seen = set()
    for record in records:
        # Nr3D annotation IDs are integers; preserve their serialized type.
        key = (record['scene_id'], str(record['object_id']), record['ann_id'])
        if key in seen:
            raise ValueError(f'Duplicate sample key: {key}')
        seen.add(key)
        scores = candidate_scores(record.get('instance_verification', {}))
        if not scores:
            continue
        best = max(scores, key=scores.get)
        scene = key[0]
        if scene not in boxes:
            with np.load(Path(npz_dir) / f'{scene}.npz', allow_pickle=True) as data:
                boxes[scene] = data['bbox8']
        bb = boxes[scene][int(best)]
        predictions[key] = {'aabb_bound': np.asarray([[bb.min(axis=0), bb.max(axis=0)]], dtype=np.float32)}
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details', required=True)
    parser.add_argument('--npz-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with open(args.details) as f:
        records = json.load(f)
    predictions = predictions_from_details(records, args.npz_dir)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as f:
        pickle.dump(predictions, f)
    print(f'{len(predictions)} predictions from {len(records)} records')


if __name__ == '__main__':
    main()
