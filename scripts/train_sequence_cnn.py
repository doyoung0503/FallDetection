#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = ["none", "occupy", "walk"]
AMPLITUDE_DIM = 114
MASK_DIM = 1
INPUT_CHANNELS = AMPLITUDE_DIM + MASK_DIM


@dataclass
class SequenceDataset:
    inputs: np.ndarray
    labels: np.ndarray
    source_csvs: list[str]
    window_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small 1D CNN baseline on CSI sequence windows."
    )
    parser.add_argument(
        "--windows-root",
        type=Path,
        default=Path("dataset/sequence_10ms_amp_mask/windows_64"),
        help="Root directory containing class-wise sequence windows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sequence_cnn_windows64"),
        help="Directory for checkpoints and training summary.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels-1", type=int, default=64)
    parser.add_argument("--hidden-channels-2", type=int, default=128)
    parser.add_argument("--kernel-size-1", type=int, default=5)
    parser.add_argument("--kernel-size-2", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stratified_file_split(
    rows_by_class: dict[str, list[dict[str, str]]],
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    train_files: dict[str, set[str]] = {}
    val_files: dict[str, set[str]] = {}

    for class_name, rows in rows_by_class.items():
        unique_files = sorted({row["source_csv"] for row in rows})
        unique_files = list(unique_files)
        rng.shuffle(unique_files)
        val_count = max(1, int(round(len(unique_files) * val_ratio)))
        val_split = set(unique_files[:val_count])
        train_split = set(unique_files[val_count:])
        train_files[class_name] = train_split
        val_files[class_name] = val_split

    return train_files, val_files


def load_window_npz(window_path: Path) -> np.ndarray:
    data = np.load(window_path)
    amplitude = data["amplitude"].astype(np.float32)
    interp_mask = data["interp_mask"].astype(np.float32)[:, None]
    stacked = np.concatenate([amplitude, interp_mask], axis=1)
    return stacked.transpose(1, 0).astype(np.float32)


def build_dataset(
    rows_by_class: dict[str, list[dict[str, str]]],
    selected_files_by_class: dict[str, set[str]],
) -> SequenceDataset:
    inputs: list[np.ndarray] = []
    labels: list[int] = []
    source_csvs: list[str] = []
    window_paths: list[str] = []

    for label, class_name in enumerate(CLASS_NAMES):
        for row in rows_by_class[class_name]:
            if row["source_csv"] not in selected_files_by_class[class_name]:
                continue

            window_path = Path(row["window_path"])
            inputs.append(load_window_npz(window_path))
            labels.append(label)
            source_csvs.append(row["source_csv"])
            window_paths.append(str(window_path))

    return SequenceDataset(
        inputs=np.stack(inputs, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        source_csvs=source_csvs,
        window_paths=window_paths,
    )


class Conv1D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rng: np.random.Generator,
    ) -> None:
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.W = rng.normal(
            0.0,
            scale,
            size=(out_channels, in_channels, kernel_size),
        ).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (self.padding, self.padding)),
            mode="constant",
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            x_pad, self.kernel_size, axis=2
        )
        out = np.einsum("bctk,ock->bot", windows, self.W) + self.b[None, :, None]
        return out, (x, windows)

    def backward(
        self, dout: np.ndarray, cache: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, windows = cache
        dx_pad = np.zeros(
            (x.shape[0], x.shape[1], x.shape[2] + 2 * self.padding), dtype=np.float32
        )
        dW = np.einsum("bot,bctk->ock", dout, windows)
        db = np.sum(dout, axis=(0, 2))

        time_len = x.shape[2]
        for kernel_idx in range(self.kernel_size):
            dx_pad[:, :, kernel_idx : kernel_idx + time_len] += np.einsum(
                "bot,oc->bct", dout, self.W[:, :, kernel_idx]
            )

        dx = dx_pad[:, :, self.padding : self.padding + time_len]
        return dx, dW.astype(np.float32), db.astype(np.float32)


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator) -> None:
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, scale, size=(in_features, out_features)).astype(
            np.float32
        )
        self.b = np.zeros(out_features, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = x @ self.W + self.b
        return out, x

    def backward(
        self, dout: np.ndarray, cache: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = cache
        dW = x.T @ dout
        db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx.astype(np.float32), dW.astype(np.float32), db.astype(np.float32)


class SequenceCNN:
    def __init__(
        self,
        input_channels: int,
        hidden_channels_1: int,
        hidden_channels_2: int,
        kernel_size_1: int,
        kernel_size_2: int,
        output_dim: int,
        rng: np.random.Generator,
    ) -> None:
        self.conv1 = Conv1D(
            in_channels=input_channels,
            out_channels=hidden_channels_1,
            kernel_size=kernel_size_1,
            rng=rng,
        )
        self.conv2 = Conv1D(
            in_channels=hidden_channels_1,
            out_channels=hidden_channels_2,
            kernel_size=kernel_size_2,
            rng=rng,
        )
        self.fc = Linear(hidden_channels_2, output_dim, rng)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1, conv1_cache = self.conv1.forward(x)
        a1 = np.maximum(z1, 0.0)
        z2, conv2_cache = self.conv2.forward(a1)
        a2 = np.maximum(z2, 0.0)
        pooled = np.mean(a2, axis=2)
        logits, fc_cache = self.fc.forward(pooled)
        cache = {
            "conv1_cache": conv1_cache,
            "z1": z1,
            "conv2_cache": conv2_cache,
            "z2": z2,
            "a2_shape": a2.shape,
            "fc_cache": fc_cache,
        }
        return logits, cache

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "conv1.W": self.conv1.W,
            "conv1.b": self.conv1.b,
            "conv2.W": self.conv2.W,
            "conv2.b": self.conv2.b,
            "fc.W": self.fc.W,
            "fc.b": self.fc.b,
        }

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.conv1.W = params["conv1.W"]
        self.conv1.b = params["conv1.b"]
        self.conv2.W = params["conv2.W"]
        self.conv2.b = params["conv2.b"]
        self.fc.W = params["fc.W"]
        self.fc.b = params["fc.b"]

    def backward(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        cache: dict[str, np.ndarray],
        weight_decay: float,
    ) -> tuple[float, dict[str, np.ndarray]]:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        batch_size = targets.shape[0]

        loss = -np.log(probs[np.arange(batch_size), targets] + 1e-12).mean()
        params = self.parameters()
        weight_keys = ["conv1.W", "conv2.W", "fc.W"]
        loss += 0.5 * weight_decay * sum(np.sum(params[key] ** 2) for key in weight_keys)

        dlogits = probs
        dlogits[np.arange(batch_size), targets] -= 1.0
        dlogits /= batch_size

        grads: dict[str, np.ndarray] = {}

        dpooled, grads["fc.W"], grads["fc.b"] = self.fc.backward(
            dlogits, cache["fc_cache"]
        )
        grads["fc.W"] += weight_decay * self.fc.W

        a2_shape = cache["a2_shape"]
        da2 = np.broadcast_to(
            (dpooled / a2_shape[2])[:, :, None], a2_shape
        ).astype(np.float32)
        dz2 = da2 * (cache["z2"] > 0.0)
        da1, grads["conv2.W"], grads["conv2.b"] = self.conv2.backward(
            dz2, cache["conv2_cache"]
        )
        grads["conv2.W"] += weight_decay * self.conv2.W

        dz1 = da1 * (cache["z1"] > 0.0)
        _, grads["conv1.W"], grads["conv1.b"] = self.conv1.backward(
            dz1, cache["conv1_cache"]
        )
        grads["conv1.W"] += weight_decay * self.conv1.W

        return float(loss), grads

    def predict(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        predictions: list[np.ndarray] = []
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            logits, _ = self.forward(x[start:end])
            predictions.append(logits.argmax(axis=1))
        return np.concatenate(predictions, axis=0)


class AdamOptimizer:
    def __init__(self, params: dict[str, np.ndarray], learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_count = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.step_count += 1
        for name, grad in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1**self.step_count)
            v_hat = self.v[name] / (1.0 - self.beta2**self.step_count)
            params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def compute_loss_and_accuracy(
    model: SequenceCNN,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    weight_decay: float,
) -> tuple[float, float]:
    losses: list[float] = []
    correct = 0
    for start in range(0, len(inputs), batch_size):
        end = start + batch_size
        batch_x = inputs[start:end]
        batch_y = targets[start:end]
        logits, cache = model.forward(batch_x)
        loss, _ = model.backward(logits, batch_y, cache, weight_decay)
        losses.append(loss)
        correct += int(np.sum(logits.argmax(axis=1) == batch_y))

    return float(np.mean(losses)), float(correct / len(inputs))


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def save_checkpoint(
    output_dir: Path,
    model: SequenceCNN,
    mean: np.ndarray,
    std: np.ndarray,
    metadata: dict[str, object],
    history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = model.parameters()
    np.savez(
        output_dir / "best_model.npz",
        conv1_W=params["conv1.W"],
        conv1_b=params["conv1.b"],
        conv2_W=params["conv2.W"],
        conv2_b=params["conv2.b"],
        fc_W=params["fc.W"],
        fc_b=params["fc.b"],
        mean=mean,
        std=std,
    )
    summary = metadata | {"history": history}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rows_by_class = {
        class_name: load_manifest_rows(args.windows_root / f"{class_name}_manifest.csv")
        for class_name in CLASS_NAMES
    }
    train_files, val_files = stratified_file_split(rows_by_class, args.val_ratio, rng)

    train_dataset = build_dataset(rows_by_class, train_files)
    val_dataset = build_dataset(rows_by_class, val_files)

    x_train = train_dataset.inputs
    y_train = train_dataset.labels
    x_val = val_dataset.inputs
    y_val = val_dataset.labels

    mean = x_train[:, :AMPLITUDE_DIM, :].mean(axis=(0, 2), keepdims=True)
    std = x_train[:, :AMPLITUDE_DIM, :].std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0

    x_train = x_train.copy()
    x_val = x_val.copy()
    x_train[:, :AMPLITUDE_DIM, :] = (x_train[:, :AMPLITUDE_DIM, :] - mean) / std
    x_val[:, :AMPLITUDE_DIM, :] = (x_val[:, :AMPLITUDE_DIM, :] - mean) / std

    model = SequenceCNN(
        input_channels=INPUT_CHANNELS,
        hidden_channels_1=args.hidden_channels_1,
        hidden_channels_2=args.hidden_channels_2,
        kernel_size_1=args.kernel_size_1,
        kernel_size_2=args.kernel_size_2,
        output_dim=len(CLASS_NAMES),
        rng=rng,
    )
    optimizer = AdamOptimizer(model.parameters(), args.learning_rate)

    best_val_acc = -1.0
    best_params = {name: value.copy() for name, value in model.parameters().items()}
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        permutation = rng.permutation(len(x_train))
        x_epoch = x_train[permutation]
        y_epoch = y_train[permutation]

        batch_losses: list[float] = []
        for start in range(0, len(x_epoch), args.batch_size):
            end = start + args.batch_size
            batch_x = x_epoch[start:end]
            batch_y = y_epoch[start:end]
            logits, cache = model.forward(batch_x)
            loss, grads = model.backward(logits, batch_y, cache, args.weight_decay)
            optimizer.step(model.parameters(), grads)
            batch_losses.append(loss)

        train_loss, train_acc = compute_loss_and_accuracy(
            model, x_train, y_train, args.batch_size, args.weight_decay
        )
        val_loss, val_acc = compute_loss_and_accuracy(
            model, x_val, y_val, args.batch_size, args.weight_decay
        )

        epoch_metrics = {
            "epoch": epoch,
            "batch_loss_mean": float(np.mean(batch_losses)),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {name: value.copy() for name, value in model.parameters().items()}

    model.set_parameters(best_params)
    val_predictions = model.predict(x_val, args.batch_size)
    val_confusion = confusion_matrix(y_val, val_predictions, len(CLASS_NAMES))

    metadata = {
        "windows_root": str(args.windows_root),
        "output_dir": str(args.output_dir),
        "class_names": CLASS_NAMES,
        "input_channels": INPUT_CHANNELS,
        "amplitude_channels": AMPLITUDE_DIM,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_channels": [args.hidden_channels_1, args.hidden_channels_2],
        "kernel_sizes": [args.kernel_size_1, args.kernel_size_2],
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "split_unit": "source_csv",
        "samples_train": int(len(x_train)),
        "samples_val": int(len(x_val)),
        "train_class_counts": {
            CLASS_NAMES[idx]: int(np.sum(y_train == idx)) for idx in range(len(CLASS_NAMES))
        },
        "val_class_counts": {
            CLASS_NAMES[idx]: int(np.sum(y_val == idx)) for idx in range(len(CLASS_NAMES))
        },
        "train_source_file_counts": {
            class_name: len(train_files[class_name]) for class_name in CLASS_NAMES
        },
        "val_source_file_counts": {
            class_name: len(val_files[class_name]) for class_name in CLASS_NAMES
        },
        "best_val_accuracy": float(best_val_acc),
        "validation_confusion_matrix": val_confusion.tolist(),
    }

    save_checkpoint(
        output_dir=args.output_dir,
        model=model,
        mean=mean.squeeze(),
        std=std.squeeze(),
        metadata=metadata,
        history=history,
    )

    print(f"saved_checkpoint={args.output_dir / 'best_model.npz'}")
    print(f"saved_summary={args.output_dir / 'training_summary.json'}")
    print("validation_confusion_matrix=")
    print(val_confusion)


if __name__ == "__main__":
    main()
