"""Fixed-size MLP baseline for the direct missing inverse action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..preconditioners import InverseAction


@dataclass(frozen=True)
class DirectMLPConfig:
    """Deliberately small fixed-size proof-of-concept architecture and trainer."""

    hidden_width: int = 128
    hidden_layers: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 32
    max_epochs: int = 250
    patience: int = 30
    seed: int = 20260829
    input_direction_scale: float = 1.0


class DirectComplementMLP(nn.Module):
    """Dense residual-and-parameter map with one scalar correction per node."""

    def __init__(
        self,
        n_nodes: int,
        n_system_features: int,
        config: DirectMLPConfig,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_width = n_nodes + n_system_features
        for _ in range(config.hidden_layers):
            layers.extend(
                [
                    nn.Linear(input_width, config.hidden_width),
                    nn.GELU(),
                ]
            )
            input_width = config.hidden_width
        layers.append(nn.Linear(input_width, n_nodes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass(frozen=True)
class DirectNormalizer:
    """Scalar normalization preserving residual/correction homogeneity."""

    residual_dimension_scale: float
    target_scale: float
    system_mean: Tuple[float, ...]
    system_scale: Tuple[float, ...]

    def encode(
        self,
        residuals: np.ndarray,
        system_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        norms = np.linalg.norm(residuals, axis=1)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("zero residual cannot be encoded")
        normalized_residuals = (
            residuals / norms[:, None] * self.residual_dimension_scale
        )
        mean = np.asarray(self.system_mean)
        scale = np.asarray(self.system_scale)
        normalized_systems = (system_features - mean) / scale
        return (
            np.column_stack((normalized_residuals, normalized_systems)),
            norms,
        )

    def encode_targets(
        self,
        corrections: np.ndarray,
        residual_norms: np.ndarray,
    ) -> np.ndarray:
        return (
            corrections
            / residual_norms[:, None]
            / self.target_scale
        )

    def decode_targets(
        self,
        normalized_corrections: np.ndarray,
        residual_norms: np.ndarray,
    ) -> np.ndarray:
        return (
            normalized_corrections
            * self.target_scale
            * residual_norms[:, None]
        )


@dataclass
class TrainingResult:
    """Trained predictor plus auditable epoch history and timing."""

    predictor: "DirectMLPPredictor"
    history: List[Dict[str, float]]
    training_seconds: float
    best_epoch: int
    best_validation_loss: float


class DirectMLPPredictor:
    """NumPy-facing, positively homogeneous wrapper around the Torch model."""

    def __init__(
        self,
        model: DirectComplementMLP,
        normalizer: DirectNormalizer,
        system_feature_names: Sequence[str],
    ) -> None:
        self.model = model.eval()
        self.normalizer = normalizer
        self.system_feature_names = tuple(system_feature_names)
        self.inference_seconds = 0.0
        self.inference_count = 0

    def reset_timing(self) -> None:
        self.inference_seconds = 0.0
        self.inference_count = 0

    def predict_batch(
        self,
        residuals: np.ndarray,
        system_features: np.ndarray,
    ) -> np.ndarray:
        residuals = np.asarray(residuals, dtype=float)
        system_features = np.asarray(system_features, dtype=float)
        encoded, norms = self.normalizer.encode(residuals, system_features)
        start = perf_counter()
        with torch.no_grad():
            prediction = (
                self.model(torch.from_numpy(encoded.astype(np.float32)))
                .cpu()
                .numpy()
                .astype(float)
            )
        self.inference_seconds += perf_counter() - start
        self.inference_count += residuals.shape[0]
        return self.normalizer.decode_targets(prediction, norms)

    def predict(
        self,
        residual: np.ndarray,
        system_features: np.ndarray,
    ) -> np.ndarray:
        return self.predict_batch(
            np.asarray(residual, dtype=float)[None, :],
            np.asarray(system_features, dtype=float)[None, :],
        )[0]

    def save(
        self,
        path: Path,
        config: DirectMLPConfig,
        history: Sequence[Dict[str, float]],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_nodes": int(self.model.network[-1].out_features),
                "system_feature_names": self.system_feature_names,
                "normalizer": asdict(self.normalizer),
                "config": asdict(config),
                "history": list(history),
            },
            path,
        )


def _build_normalizer(
    train_residuals: np.ndarray,
    train_corrections: np.ndarray,
    train_system_features: np.ndarray,
    input_direction_scale: float = 1.0,
) -> DirectNormalizer:
    residual_norms = np.linalg.norm(train_residuals, axis=1)
    per_unit_correction = train_corrections / residual_norms[:, None]
    target_scale = float(np.sqrt(np.mean(per_unit_correction**2)))
    if not np.isfinite(target_scale) or target_scale <= 0.0:
        raise ValueError("training corrections have invalid scale")
    system_mean = np.mean(train_system_features, axis=0)
    system_scale = np.std(train_system_features, axis=0)
    system_scale[system_scale <= np.finfo(float).eps] = 1.0
    return DirectNormalizer(
        residual_dimension_scale=float(input_direction_scale),
        target_scale=target_scale,
        system_mean=tuple(float(value) for value in system_mean),
        system_scale=tuple(float(value) for value in system_scale),
    )


def train_direct_mlp(
    train_residuals: np.ndarray,
    train_corrections: np.ndarray,
    train_system_features: np.ndarray,
    validation_residuals: np.ndarray,
    validation_corrections: np.ndarray,
    validation_system_features: np.ndarray,
    system_feature_names: Sequence[str],
    config: DirectMLPConfig,
) -> TrainingResult:
    r"""Train N_theta(r, xi) against A(xi)^-1 r - M(xi)^-1 r."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    normalizer = _build_normalizer(
        train_residuals,
        train_corrections,
        train_system_features,
        config.input_direction_scale,
    )
    train_inputs, train_norms = normalizer.encode(
        train_residuals, train_system_features
    )
    validation_inputs, validation_norms = normalizer.encode(
        validation_residuals, validation_system_features
    )
    train_targets = normalizer.encode_targets(
        train_corrections, train_norms
    )
    validation_targets = normalizer.encode_targets(
        validation_corrections, validation_norms
    )

    dataset = TensorDataset(
        torch.from_numpy(train_inputs.astype(np.float32)),
        torch.from_numpy(train_targets.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = DirectComplementMLP(
        train_residuals.shape[1],
        train_system_features.shape[1],
        config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    validation_input_tensor = torch.from_numpy(
        validation_inputs.astype(np.float32)
    )
    validation_target_tensor = torch.from_numpy(
        validation_targets.astype(np.float32)
    )

    history: List[Dict[str, float]] = []
    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(
                    model(validation_input_tensor),
                    validation_target_tensor,
                )
            )
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation - 1.0e-7:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_seconds = perf_counter() - start
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation state")
    model.load_state_dict(best_state)
    predictor = DirectMLPPredictor(
        model,
        normalizer,
        system_feature_names,
    )
    return TrainingResult(
        predictor,
        history,
        training_seconds,
        best_epoch,
        best_validation,
    )


def correction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return relative error, magnitude ratio, and cosine per sample."""

    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    target_norms = np.linalg.norm(targets, axis=1)
    prediction_norms = np.linalg.norm(predictions, axis=1)
    relative = np.linalg.norm(predictions - targets, axis=1) / np.maximum(
        target_norms, 1.0e-300
    )
    cosine = np.sum(predictions * targets, axis=1) / np.maximum(
        prediction_norms * target_norms, 1.0e-300
    )
    magnitude = prediction_norms / np.maximum(target_norms, 1.0e-300)
    return {
        "relative_error": relative,
        "magnitude_ratio": magnitude,
        "cosine_similarity": cosine,
    }


def make_direct_preconditioner(
    base: InverseAction,
    predictor: DirectMLPPredictor,
    system_features: np.ndarray,
    trace_inputs: Optional[List[np.ndarray]] = None,
    trace_limit: int = 0,
) -> InverseAction:
    r"""Return P_dir(r)=M^-1 r + N_theta(r, xi) as a nonlinear action."""

    features = np.asarray(system_features, dtype=float)

    def action(residual: np.ndarray) -> np.ndarray:
        if trace_inputs is not None and len(trace_inputs) < trace_limit:
            trace_inputs.append(np.asarray(residual, dtype=float).copy())
        return base.apply(residual) + predictor.predict(residual, features)

    return InverseAction(
        f"{base.name}+direct_mlp",
        action,
        setup_seconds=base.setup_seconds,
        is_linear=False,
        diagnostics={"learned_strategy": "direct_missing_action"},
    )
