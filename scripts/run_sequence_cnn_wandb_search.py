#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


WINDOWS_50_CONFIGS = [
    {
        "name": "adam_base",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "hidden_channels_1": 64,
        "hidden_channels_2": 128,
        "kernel_size_1": 5,
        "kernel_size_2": 3,
    },
    {
        "name": "adam_lr5e4_k73",
        "optimizer": "adam",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "hidden_channels_1": 64,
        "hidden_channels_2": 128,
        "kernel_size_1": 7,
        "kernel_size_2": 3,
    },
    {
        "name": "adam_smallbatch_wide",
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "weight_decay": 5e-5,
        "batch_size": 32,
        "hidden_channels_1": 96,
        "hidden_channels_2": 192,
        "kernel_size_1": 5,
        "kernel_size_2": 3,
    },
    {
        "name": "adam_deeperreg",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "hidden_channels_1": 128,
        "hidden_channels_2": 256,
        "kernel_size_1": 5,
        "kernel_size_2": 5,
    },
    {
        "name": "sam_reference",
        "optimizer": "sam_sgd",
        "learning_rate": 1e-2,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "hidden_channels_1": 64,
        "hidden_channels_2": 128,
        "kernel_size_1": 5,
        "kernel_size_2": 3,
        "sam_rho": 0.05,
        "sgd_momentum": 0.9,
    },
]


WINDOWS_64_CONFIGS = [
    {
        "name": "adam_base",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "hidden_channels_1": 64,
        "hidden_channels_2": 128,
        "kernel_size_1": 5,
        "kernel_size_2": 3,
    },
    {
        "name": "adam_lr5e4_k75",
        "optimizer": "adam",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "hidden_channels_1": 64,
        "hidden_channels_2": 128,
        "kernel_size_1": 7,
        "kernel_size_2": 5,
    },
    {
        "name": "adam_smallbatch_wide",
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "weight_decay": 5e-5,
        "batch_size": 32,
        "hidden_channels_1": 96,
        "hidden_channels_2": 192,
        "kernel_size_1": 5,
        "kernel_size_2": 3,
    },
    {
        "name": "adam_highcapacity",
        "optimizer": "adam",
        "learning_rate": 7e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "hidden_channels_1": 128,
        "hidden_channels_2": 256,
        "kernel_size_1": 5,
        "kernel_size_2": 5,
    },
    {
        "name": "sam_reference",
        "optimizer": "sam_sgd",
        "learning_rate": 1e-2,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "hidden_channels_1": 64,
        "hidden_channels_2": 128,
        "kernel_size_1": 5,
        "kernel_size_2": 3,
        "sam_rho": 0.05,
        "sgd_momentum": 0.9,
    },
]


DATASET_CONFIGS = {
    "windows_50": WINDOWS_50_CONFIGS,
    "windows_64": WINDOWS_64_CONFIGS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed 1D CNN hyperparameter searches for windows_50 and windows_64 "
            "and log each run to Weights & Biases."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset/, artifacts/, and scripts/.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch training runs.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="fall-detection-sequence-cnn-search",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Device forwarded to the training script.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epoch count for every run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for every run.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("time_file", "random_file", "time_block"),
        default="time_file",
        help="Data split mode forwarded to the training script.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count forwarded to the training script.",
    )
    parser.add_argument(
        "--smoke-train-limit",
        type=int,
        default=None,
        help="Optional per-class train window cap for quick smoke tests.",
    )
    parser.add_argument(
        "--smoke-val-limit",
        type=int,
        default=None,
        help="Optional per-class validation window cap for quick smoke tests.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["windows_50", "windows_64"],
        choices=sorted(DATASET_CONFIGS),
        help="Dataset variants to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs to execute, useful for smoke tests.",
    )
    return parser.parse_args()


def build_command(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    config: dict[str, object],
) -> tuple[list[str], Path]:
    project_root = args.project_root.resolve()
    windows_root = project_root / "dataset" / "sequence_10ms_amp_mask" / dataset_name
    run_name = f"{dataset_name}_{config['name']}"
    output_dir = project_root / "artifacts" / "wandb_search" / dataset_name / run_name
    train_script = project_root / "scripts" / "train_sequence_cnn_torch.py"

    command = [
        args.python_bin,
        str(train_script),
        "--windows-root",
        str(windows_root),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(config["batch_size"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--optimizer",
        str(config["optimizer"]),
        "--hidden-channels-1",
        str(config["hidden_channels_1"]),
        "--hidden-channels-2",
        str(config["hidden_channels_2"]),
        "--kernel-size-1",
        str(config["kernel_size_1"]),
        "--kernel-size-2",
        str(config["kernel_size_2"]),
        "--split-mode",
        args.split_mode,
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-run-name",
        run_name,
        "--wandb-group",
        dataset_name,
        "--wandb-job-type",
        "hparam-search",
        "--wandb-tags",
        "sequence-cnn",
        dataset_name,
        "manual-grid",
    ]

    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.smoke_train_limit is not None:
        command.extend(["--smoke-train-limit", str(args.smoke_train_limit)])
    if args.smoke_val_limit is not None:
        command.extend(["--smoke-val-limit", str(args.smoke_val_limit)])
    if config["optimizer"] == "sam_sgd":
        command.extend(["--sam-rho", str(config["sam_rho"])])
        command.extend(["--sgd-momentum", str(config["sgd_momentum"])])

    return command, output_dir


def main() -> None:
    args = parse_args()
    launch_records: list[dict[str, object]] = []
    executed_runs = 0

    for dataset_name in args.datasets:
        configs = DATASET_CONFIGS[dataset_name]
        for config in configs:
            if args.max_runs is not None and executed_runs >= args.max_runs:
                break
            command, output_dir = build_command(
                args,
                dataset_name=dataset_name,
                config=config,
            )

            record = {
                "dataset_name": dataset_name,
                "run_name": f"{dataset_name}_{config['name']}",
                "output_dir": str(output_dir),
                "command": command,
                "config": config,
            }
            launch_records.append(record)

            print(f"\n=== {record['run_name']} ===")
            print(" ".join(command))
            if args.dry_run:
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=str(args.project_root.resolve()))
            executed_runs += 1
        if args.max_runs is not None and executed_runs >= args.max_runs:
            break

    launch_summary = {
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "device": args.device,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_mode": args.split_mode,
        "datasets": args.datasets,
        "runs": launch_records,
    }
    summary_path = (
        args.project_root.resolve()
        / "artifacts"
        / "wandb_search"
        / "launch_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(launch_summary, indent=2))
    print(f"\nsaved_launch_summary={summary_path}")


if __name__ == "__main__":
    main()
