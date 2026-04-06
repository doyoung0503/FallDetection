#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "esp32_sequence_htltf_ma10diff_w64_s10_tol4000"
    / "ma10_diff"
    / "windows_64"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "esp32_sequence_ma10_cnn_mps"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_esp32_sequence_cnn_torch.py"
SEEDS = [42, 43, 44]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3-seed 1D CNN experiments on the moving-average residual dataset."
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to launch training.",
    )
    parser.add_argument(
        "--variant-root",
        type=Path,
        default=DEFAULT_VARIANT_ROOT,
        help="Root directory such as .../ma10_diff/windows_64.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for run outputs and aggregated summary.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels-1", type=int, default=64)
    parser.add_argument("--hidden-channels-2", type=int, default=128)
    parser.add_argument("--kernel-size-1", type=int, default=5)
    parser.add_argument("--kernel-size-2", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="mps",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    per_seed_results: list[dict[str, object]] = []
    for seed in SEEDS:
        output_dir = args.output_root / f"seed_{seed}"
        command = [
            str(args.python_bin),
            str(TRAIN_SCRIPT),
            "--variant-root",
            str(args.variant_root),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--hidden-channels-1",
            str(args.hidden_channels_1),
            "--hidden-channels-2",
            str(args.hidden_channels_2),
            "--kernel-size-1",
            str(args.kernel_size_1),
            "--kernel-size-2",
            str(args.kernel_size_2),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--num-workers",
            str(args.num_workers),
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        per_seed_results.append(
            json.loads((output_dir / "training_summary.json").read_text())
        )

    accuracy_values = [float(run["best_validation_accuracy"]) for run in per_seed_results]
    macro_f1_values = [float(run["metrics"]["macro_f1"]) for run in per_seed_results]
    loss_values = [float(run["best_validation_loss"]) for run in per_seed_results]
    epoch_values = [float(run["best_epoch"]) for run in per_seed_results]

    aggregate = {
        "config": {
            "variant_root": str(args.variant_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_channels_1": args.hidden_channels_1,
            "hidden_channels_2": args.hidden_channels_2,
            "kernel_size_1": args.kernel_size_1,
            "kernel_size_2": args.kernel_size_2,
            "device": args.device,
            "seeds": SEEDS,
        },
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
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
