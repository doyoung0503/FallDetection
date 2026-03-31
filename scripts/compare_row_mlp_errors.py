#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_row_mlp import (
    CLASS_NAMES,
    INPUT_DIM,
    MLPClassifier,
    load_dataset,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare misclassified validation samples across saved row-MLP checkpoints."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/preprocessed_raw"),
        help="Root directory containing none/occupy/walk CSV folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("analysis/row_mlp_error_comparison.json"),
        help="Where to save the comparison report.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "CHECKPOINT"),
        required=True,
        help="Model name and checkpoint path (.npz). Repeat this flag for multiple models.",
    )
    return parser.parse_args()


def load_checkpoint_model(checkpoint_path: Path) -> tuple[MLPClassifier, np.ndarray, np.ndarray]:
    checkpoint = np.load(checkpoint_path)

    hidden_dim_1 = int(checkpoint["W1"].shape[1])
    hidden_dim_2 = int(checkpoint["W2"].shape[1])
    output_dim = int(checkpoint["W3"].shape[1])

    model = MLPClassifier(
        input_dim=INPUT_DIM,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        output_dim=output_dim,
        rng=np.random.default_rng(0),
    )
    model.params = {
        "W1": checkpoint["W1"],
        "b1": checkpoint["b1"],
        "W2": checkpoint["W2"],
        "b2": checkpoint["b2"],
        "W3": checkpoint["W3"],
        "b3": checkpoint["b3"],
    }
    mean = checkpoint["mean"].reshape(1, -1)
    std = checkpoint["std"].reshape(1, -1)
    return model, mean, std


def sample_record(
    dataset_bundle,
    original_index: int,
    y_val: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    val_idx_map: dict[int, int],
) -> dict[str, object]:
    val_position = val_idx_map[original_index]
    record = {
        "dataset_index": int(original_index),
        "path": dataset_bundle.sample_paths[original_index],
        "row_number": int(dataset_bundle.row_numbers[original_index]),
        "true_label": CLASS_NAMES[int(y_val[val_position])],
        "predictions": {
            model_name: CLASS_NAMES[int(predictions[val_position])]
            for model_name, predictions in predictions_by_model.items()
        },
    }
    return record


def main() -> None:
    args = parse_args()

    bundle = load_dataset(args.dataset_root, max_samples_per_class=None)
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx = stratified_split(bundle.labels, args.val_ratio, rng)

    x_val_raw = bundle.features[val_idx]
    y_val = bundle.labels[val_idx]
    val_idx_map = {int(original_idx): position for position, original_idx in enumerate(val_idx)}

    predictions_by_model: dict[str, np.ndarray] = {}
    misclassified_sets: dict[str, set[int]] = {}
    per_model_summary: dict[str, object] = {}

    for model_name, checkpoint_str in args.model:
        checkpoint_path = Path(checkpoint_str)
        model, mean, std = load_checkpoint_model(checkpoint_path)
        x_val = (x_val_raw - mean) / std
        predictions = model.predict(x_val)
        predictions_by_model[model_name] = predictions

        wrong_positions = np.where(predictions != y_val)[0]
        wrong_dataset_indices = {int(val_idx[position]) for position in wrong_positions}
        misclassified_sets[model_name] = wrong_dataset_indices

        per_model_summary[model_name] = {
            "checkpoint": str(checkpoint_path),
            "misclassified_count": int(len(wrong_positions)),
            "correct_count": int(len(y_val) - len(wrong_positions)),
        }

    model_names = list(predictions_by_model.keys())
    pairwise_overlap: list[dict[str, object]] = []
    for left_idx in range(len(model_names)):
        for right_idx in range(left_idx + 1, len(model_names)):
            left_name = model_names[left_idx]
            right_name = model_names[right_idx]
            left_set = misclassified_sets[left_name]
            right_set = misclassified_sets[right_name]
            intersection = left_set & right_set
            union = left_set | right_set
            pairwise_overlap.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "left_count": len(left_set),
                    "right_count": len(right_set),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "jaccard": float(len(intersection) / len(union)) if union else 1.0,
                }
            )

    common_wrong = set.intersection(*misclassified_sets.values()) if misclassified_sets else set()

    unique_wrong_by_model = {}
    for model_name in model_names:
        other_sets = [misclassified_sets[name] for name in model_names if name != model_name]
        others_union = set().union(*other_sets) if other_sets else set()
        unique_wrong = misclassified_sets[model_name] - others_union
        unique_wrong_by_model[model_name] = [
            sample_record(bundle, dataset_index, y_val, predictions_by_model, val_idx_map)
            for dataset_index in sorted(unique_wrong)
        ]

    report = {
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "validation_sample_count": int(len(val_idx)),
        "models": per_model_summary,
        "pairwise_overlap": pairwise_overlap,
        "common_misclassified_count": len(common_wrong),
        "common_misclassified_samples": [
            sample_record(bundle, dataset_index, y_val, predictions_by_model, val_idx_map)
            for dataset_index in sorted(common_wrong)
        ],
        "unique_misclassified_samples": unique_wrong_by_model,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2))
    print(f"saved_report={args.output_path}")
    print(f"common_misclassified_count={len(common_wrong)}")
    for model_name in model_names:
        print(f"{model_name}_misclassified_count={len(misclassified_sets[model_name])}")


if __name__ == "__main__":
    main()
