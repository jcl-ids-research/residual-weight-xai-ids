"""Deep-model supplement to the V3 class-imbalance study.

The runner reuses the V3 split, preprocessing, resampling, weighting, and
evaluation code without modification, so every metric it emits is directly
comparable with the XGBoost and tree-ensemble results already reported. Its
purpose is to test whether post-cleaning residual weighting transfers to
neural models, and to supply the deep baseline the manuscript currently lacks.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_v3_explainable import (
    build_resampled_sets,
    evaluate,
    inverse_frequency_weights,
    json_default,
    sample_weights,
)
from v3_data import (
    LOADERS,
    TrainOnlyPreprocessor,
    moderate_targets,
    split_official_training,
)

CONFIGURATIONS: Final = (
    "raw",
    "smotenc_tomek_weight",
    "smotenc_tomek_residual_weight",
)
ARCHITECTURES: Final = ("MLP",)
BATCH_SIZE: Final = 1024
MAX_EPOCHS: Final = 200
PATIENCE: Final = 20
LEARNING_RATE: Final = 1e-3
WEIGHT_DECAY: Final = 1e-4


class MLP(nn.Module):
    """Standard deep tabular baseline."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.network(batch)


class CNN1D(nn.Module):
    """One-dimensional convolutional baseline over the flow feature vector."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.1), nn.Linear(128, n_classes)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(batch.unsqueeze(1)))


def build_model(name: str, n_features: int, n_classes: int) -> nn.Module:
    """Instantiate one architecture by name."""
    if name == "MLP":
        return MLP(n_features, n_classes)
    if name == "CNN1D":
        return CNN1D(n_features, n_classes)
    raise ValueError(f"Unknown architecture: {name}")


def standardise(
    train: np.ndarray, others: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Scale features using training statistics only."""
    mean = train.mean(axis=0, keepdims=True)
    deviation = train.std(axis=0, keepdims=True)
    deviation[deviation < 1e-8] = 1.0
    scaled_train = ((train - mean) / deviation).astype(np.float32)
    scaled_others = tuple(
        ((item - mean) / deviation).astype(np.float32) for item in others
    )
    return scaled_train, scaled_others


def resolve_device(requested: str, required_mib: int = 3072) -> torch.device:
    """Select CUDA only when enough free memory is available, otherwise CPU.

    The host is shared, so a full GPU must degrade to CPU instead of aborting.
    """
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes / (1024 * 1024) < required_mib:
        return torch.device("cpu")
    return torch.device(requested)


def validation_loss(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute mean validation loss in batches to bound peak memory."""
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            batch_features = torch.from_numpy(features[start : start + 4096]).to(device)
            batch_labels = torch.from_numpy(
                labels[start : start + 4096].astype(np.int64)
            ).to(device)
            total += float(criterion(model(batch_features), batch_labels).sum().item())
            seen += len(batch_labels)
    return total / max(1, seen)


def predict_probability(
    model: nn.Module, features: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return class probabilities for a feature matrix."""
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            batch = torch.from_numpy(features[start : start + 4096]).to(device)
            outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probability = np.concatenate(outputs, axis=0).astype(np.float64)
    return probability / probability.sum(axis=1, keepdims=True)


def train_model(
    model: nn.Module,
    tensors: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    max_epochs: int,
) -> dict[str, float]:
    """Train with weighted cross-entropy and validation-loss early stopping."""
    features, labels, weights = tensors
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels.astype(np.int64)),
            torch.from_numpy(weights.astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(reduction="none")
    best_loss = float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    waited = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_features, batch_labels, batch_weights in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_weights = batch_weights.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            losses = criterion(model(batch_features), batch_labels)
            (losses * batch_weights).mean().backward()
            optimiser.step()
        epoch_loss = validation_loss(
            model, validation[0], validation[1], criterion, device
        )
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            waited = 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return {"best_validation_loss": best_loss, "best_epoch": best_epoch}


def run(args: argparse.Namespace) -> Path:
    """Execute every architecture and weighting configuration for one seed."""
    output = Path(args.out) / args.dataset / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {metrics_path}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    data_dirs = {
        "unsw": args.unsw_dir,
        "nslkdd": args.nslkdd_dir,
        "cicids2017": args.cicids2017_dir,
    }
    raw = LOADERS[args.dataset](data_dirs[args.dataset], smoke=args.smoke)
    n_classes = len(raw.class_names)
    normal_index = raw.class_names.index(raw.normal_class_name)
    X_sub_frame, X_val_frame, y_sub, y_val = split_official_training(raw, args.seed)

    preprocessor = TrainOnlyPreprocessor(raw.categorical_columns).fit(X_sub_frame)
    X_sub_encoded = preprocessor.encode(X_sub_frame)
    X_val = preprocessor.classifier_matrix(preprocessor.encode(X_val_frame))
    X_test = preprocessor.classifier_matrix(preprocessor.encode(raw.X_test))

    strategy = moderate_targets(y_sub, n_classes, expansion_cap=15)
    resampled, _ = build_resampled_sets(
        X_sub_encoded,
        y_sub,
        preprocessor.categorical_indices,
        strategy,
        args.seed,
        args.n_jobs,
    )
    resampled["smotenc_tomek_weight"] = resampled["smotenc_tomek"]
    resampled["smotenc_tomek_residual_weight"] = resampled["smotenc_tomek"]
    original_weights = inverse_frequency_weights(y_sub, n_classes)
    residual_weights = inverse_frequency_weights(
        resampled["smotenc_tomek"][1], n_classes
    )

    max_epochs = 3 if args.smoke else MAX_EPOCHS
    results: dict[str, dict] = {}
    for architecture in ARCHITECTURES:
        for configuration in CONFIGURATIONS:
            encoded_train, y_train = resampled[configuration]
            X_train = preprocessor.classifier_matrix(encoded_train)
            scaled_train, (scaled_val, scaled_test) = standardise(
                X_train, (X_val, X_test)
            )
            if configuration == "smotenc_tomek_weight":
                weights = sample_weights(y_train, original_weights)
                weight_source = "original_sub_training_distribution"
            elif configuration == "smotenc_tomek_residual_weight":
                weights = sample_weights(y_train, residual_weights)
                weight_source = "post_smote_tomek_distribution"
            else:
                weights = np.ones(len(y_train), dtype=np.float32)
                weight_source = "none"

            torch.manual_seed(args.seed)
            model = build_model(architecture, scaled_train.shape[1], n_classes).to(device)
            started = time.perf_counter()
            training_audit = train_model(
                model,
                (scaled_train, y_train, weights),
                (scaled_val, y_val),
                device,
                max_epochs,
            )
            train_seconds = time.perf_counter() - started
            validation_probability = predict_probability(model, scaled_val, device)
            started = time.perf_counter()
            probability = predict_probability(model, scaled_test, device)
            test_seconds = time.perf_counter() - started
            results[f"{architecture}::{configuration}"] = {
                "architecture": architecture,
                "configuration": configuration,
                "weight_source": weight_source,
                "training_rows": int(len(y_train)),
                "metrics": evaluate(
                    y_val,
                    validation_probability,
                    raw.y_test,
                    probability.argmax(axis=1),
                    probability,
                    raw.class_names,
                    normal_index,
                ),
                "train_seconds": train_seconds,
                "test_seconds": test_seconds,
                **training_audit,
            }

    payload = {
        "schema": "v4_deep_baseline",
        "dataset": args.dataset,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "class_names": raw.class_names,
        "normal_class": raw.normal_class_name,
        "architectures": list(ARCHITECTURES),
        "configurations": list(CONFIGURATIONS),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "max_epochs": max_epochs,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.cuda.is_available(),
        },
        "runs": results,
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    return metrics_path


def parse_args() -> argparse.Namespace:
    """Parse command-line configuration."""
    parser = argparse.ArgumentParser(description="V4 deep baseline runner")
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=8)
    parser.add_argument("--unsw-dir", dest="unsw_dir", default="/opt/UNSW-NB15")
    parser.add_argument("--nslkdd-dir", dest="nslkdd_dir", default="/opt/NSL-KDD")
    parser.add_argument(
        "--cicids2017-dir",
        dest="cicids2017_dir",
        default="/opt/CIC-IDS-2017/MachineLearningCVE",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps({"metrics": str(run(parse_args()))}, ensure_ascii=False))
