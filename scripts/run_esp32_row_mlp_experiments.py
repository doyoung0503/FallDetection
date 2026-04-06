#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "esp32_row_mlp_mps"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_esp32_row_mlp_torch.py"

VARIANTS = ["lltf_only", "htltf_only", "lltf_htltf"]
SEEDS = [42, 43, 44]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3-seed row-MLP experiments on all esp32 raw CSI variants."
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to launch the training script.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing lltf_only/, htltf_only/, lltf_htltf/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where all run outputs and aggregated summaries are stored.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim-1", type=int, default=256)
    parser.add_argument("--hidden-dim-2", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="mps",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def summarize_run(summary_path: Path) -> dict[str, object]:
    return json.loads(summary_path.read_text())


def mean_std(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, object] = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim_1": args.hidden_dim_1,
            "hidden_dim_2": args.hidden_dim_2,
            "device": args.device,
            "seeds": SEEDS,
        },
        "variants": {},
    }

    for variant in VARIANTS:
        variant_root = args.dataset_root / variant
        variant_output_root = args.output_root / variant
        variant_output_root.mkdir(parents=True, exist_ok=True)

        per_seed_results: list[dict[str, object]] = []
        for seed in SEEDS:
            run_output_dir = variant_output_root / f"seed_{seed}"
            command = [
                str(args.python_bin),
                str(TRAIN_SCRIPT),
                "--dataset-root",
                str(variant_root),
                "--output-dir",
                str(run_output_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--hidden-dim-1",
                str(args.hidden_dim_1),
                "--hidden-dim-2",
                str(args.hidden_dim_2),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
            ]
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            summary = summarize_run(run_output_dir / "training_summary.json")
            per_seed_results.append(summary)

        accuracy_values = [float(run["best_validation_accuracy"]) for run in per_seed_results]
        macro_f1_values = [float(run["metrics"]["macro_f1"]) for run in per_seed_results]
        loss_values = [float(run["best_validation_loss"]) for run in per_seed_results]
        epoch_values = [float(run["best_epoch"]) for run in per_seed_results]

        aggregate["variants"][variant] = {
            "per_seed": [
                {
                    "seed": int(run["seed"]),
                    "best_epoch": int(run["best_epoch"]),
                    "best_validation_accuracy": float(run["best_validation_accuracy"]),
                    "best_validation_loss": float(run["best_validation_loss"]),
                    "macro_f1": float(run["metrics"]["macro_f1"]),
                    "confusion_matrix": run["best_confusion_matrix"],
                }
                for run in per_seed_results
            ],
            "accuracy": mean_std(accuracy_values),
            "macro_f1": mean_std(macro_f1_values),
            "best_validation_loss": mean_std(loss_values),
            "best_epoch": mean_std(epoch_values),
        }

    summary_path = args.output_root / "aggregate_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
