#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sam_optimizer import SAMOptimizer, SGDOptimizer


CLASS_NAMES = ["none", "occupy", "walk"]
INPUT_DIM = 228


@dataclass
class DatasetBundle:
    features: np.ndarray
    labels: np.ndarray
    sample_paths: list[str]
    row_numbers: list[int]
    skipped_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 3-layer MLP on preprocessed CSI rows."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/preprocessed_raw"),
        help="Root directory containing none/occupy/walk CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mlp_row_split"),
        help="Directory for checkpoints and training summary.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim-1", type=int, default=256)
    parser.add_argument("--hidden-dim-2", type=int, default=128)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("adam", "sam_sgd"),
        default="adam",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=0.05,
        help="Neighborhood size for SAM when --optimizer=sam_sgd.",
    )
    parser.add_argument(
        "--sgd-momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD when --optimizer=sam_sgd.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def load_dataset(dataset_root: Path, max_samples_per_class: int | None) -> DatasetBundle:
    features: list[np.ndarray] = []
    labels: list[int] = []
    sample_paths: list[str] = []
    row_numbers: list[int] = []
    skipped_rows = 0

    for label, class_name in enumerate(CLASS_NAMES):
        class_count = 0
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for csv_path in sorted(class_dir.glob("*.csv")):
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, start=2):
                    if max_samples_per_class is not None and class_count >= max_samples_per_class:
                        break

                    if row.get("len") != str(INPUT_DIM):
                        skipped_rows += 1
                        continue

                    values = ast.literal_eval(row["data"])
                    if len(values) != INPUT_DIM:
                        skipped_rows += 1
                        continue

                    features.append(np.asarray(values, dtype=np.float32))
                    labels.append(label)
                    sample_paths.append(str(csv_path))
                    row_numbers.append(row_number)
                    class_count += 1

            if max_samples_per_class is not None and class_count >= max_samples_per_class:
                break

    return DatasetBundle(
        features=np.stack(features, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        sample_paths=sample_paths,
        row_numbers=row_numbers,
        skipped_rows=skipped_rows,
    )


def stratified_split(
    labels: np.ndarray, val_ratio: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for class_id in range(len(CLASS_NAMES)):
        class_indices = np.where(labels == class_id)[0]
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_indices.append(shuffled[:val_count])
        train_indices.append(shuffled[val_count:])

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class MLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        output_dim: int,
        rng: np.random.Generator,
    ) -> None:
        self.params = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim_1)).astype(np.float32),
            "b1": np.zeros(hidden_dim_1, dtype=np.float32),
            "W2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim_1), size=(hidden_dim_1, hidden_dim_2)).astype(np.float32),
            "b2": np.zeros(hidden_dim_2, dtype=np.float32),
            "W3": rng.normal(0.0, np.sqrt(2.0 / hidden_dim_2), size=(hidden_dim_2, output_dim)).astype(np.float32),
            "b3": np.zeros(output_dim, dtype=np.float32),
        }

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = inputs @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.maximum(z2, 0.0)
        logits = a2 @ self.params["W3"] + self.params["b3"]
        cache = {"inputs": inputs, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return logits, cache

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
        loss += 0.5 * weight_decay * sum(
            np.sum(self.params[name] ** 2) for name in ("W1", "W2", "W3")
        )

        dlogits = probs
        dlogits[np.arange(batch_size), targets] -= 1.0
        dlogits /= batch_size

        grads: dict[str, np.ndarray] = {}
        grads["W3"] = cache["a2"].T @ dlogits + weight_decay * self.params["W3"]
        grads["b3"] = dlogits.sum(axis=0)

        da2 = dlogits @ self.params["W3"].T
        dz2 = da2 * (cache["z2"] > 0.0)
        grads["W2"] = cache["a1"].T @ dz2 + weight_decay * self.params["W2"]
        grads["b2"] = dz2.sum(axis=0)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (cache["z1"] > 0.0)
        grads["W1"] = cache["inputs"].T @ dz1 + weight_decay * self.params["W1"]
        grads["b1"] = dz1.sum(axis=0)
        return loss, grads

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(inputs)
        return logits.argmax(axis=1)


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
    model: MLPClassifier,
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
        preds = logits.argmax(axis=1)
        correct += int(np.sum(preds == batch_y))

    return float(np.mean(losses)), float(correct / len(inputs))


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def save_checkpoint(
    output_dir: Path,
    model: MLPClassifier,
    mean: np.ndarray,
    std: np.ndarray,
    history: list[dict[str, float]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.npz"
    np.savez(
        checkpoint_path,
        W1=model.params["W1"],
        b1=model.params["b1"],
        W2=model.params["W2"],
        b2=model.params["b2"],
        W3=model.params["W3"],
        b3=model.params["b3"],
        mean=mean,
        std=std,
    )

    summary_path = output_dir / "training_summary.json"
    summary = metadata | {"history": history}
    summary_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    bundle = load_dataset(args.dataset_root, args.max_samples_per_class)
    train_idx, val_idx = stratified_split(bundle.labels, args.val_ratio, rng)

    x_train = bundle.features[train_idx]
    y_train = bundle.labels[train_idx]
    x_val = bundle.features[val_idx]
    y_val = bundle.labels[val_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    model = MLPClassifier(
        input_dim=INPUT_DIM,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        output_dim=len(CLASS_NAMES),
        rng=rng,
    )
    optimizer: AdamOptimizer | SAMOptimizer
    if args.optimizer == "sam_sgd":
        base_optimizer = SGDOptimizer(
            learning_rate=args.learning_rate, momentum=args.sgd_momentum
        )
        optimizer = SAMOptimizer(
            params=model.params, base_optimizer=base_optimizer, rho=args.sam_rho
        )
    else:
        optimizer = AdamOptimizer(model.params, args.learning_rate)

    best_val_acc = -1.0
    best_params = {name: value.copy() for name, value in model.params.items()}
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        permutation = rng.permutation(len(x_train))
        x_train_epoch = x_train[permutation]
        y_train_epoch = y_train[permutation]

        batch_losses: list[float] = []
        for start in range(0, len(x_train_epoch), args.batch_size):
            end = start + args.batch_size
            batch_x = x_train_epoch[start:end]
            batch_y = y_train_epoch[start:end]
            logits, cache = model.forward(batch_x)
            loss, grads = model.backward(logits, batch_y, cache, args.weight_decay)

            if args.optimizer == "sam_sgd":
                optimizer.first_step(model.params, grads)

                perturbed_logits, perturbed_cache = model.forward(batch_x)
                _, perturbed_grads = model.backward(
                    perturbed_logits, batch_y, perturbed_cache, args.weight_decay
                )
                optimizer.second_step(model.params, perturbed_grads)
            else:
                optimizer.step(model.params, grads)

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
            best_params = {name: value.copy() for name, value in model.params.items()}

    model.params = best_params
    val_predictions = model.predict(x_val)
    val_confusion = confusion_matrix(y_val, val_predictions, len(CLASS_NAMES))

    metadata = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "class_names": CLASS_NAMES,
        "input_dim": INPUT_DIM,
        "hidden_dims": [args.hidden_dim_1, args.hidden_dim_2],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "sam_rho": args.sam_rho if args.optimizer == "sam_sgd" else None,
        "sgd_momentum": args.sgd_momentum if args.optimizer == "sam_sgd" else None,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "samples_total": int(len(bundle.features)),
        "samples_train": int(len(train_idx)),
        "samples_val": int(len(val_idx)),
        "skipped_rows": int(bundle.skipped_rows),
        "best_val_accuracy": float(best_val_acc),
        "validation_confusion_matrix": val_confusion.tolist(),
        "train_class_counts": {
            CLASS_NAMES[class_id]: int(np.sum(y_train == class_id))
            for class_id in range(len(CLASS_NAMES))
        },
        "val_class_counts": {
            CLASS_NAMES[class_id]: int(np.sum(y_val == class_id))
            for class_id in range(len(CLASS_NAMES))
        },
    }

    save_checkpoint(args.output_dir, model, mean.squeeze(0), std.squeeze(0), history, metadata)
    print(f"saved_checkpoint={args.output_dir / 'best_model.npz'}")
    print(f"saved_summary={args.output_dir / 'training_summary.json'}")
    print("validation_confusion_matrix=")
    print(val_confusion)


if __name__ == "__main__":
    main()
