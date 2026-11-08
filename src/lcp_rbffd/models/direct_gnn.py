"""Small message-passing baseline for the direct missing inverse action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..preconditioners import InverseAction
from ..rbffd import RBFSystem
from .direct_mlp import DirectNormalizer


@dataclass(frozen=True)
class RBFGraph:
    """RBF-FD stencil graph with row-oriented directed edges ``j -> i``."""

    label: str
    nodes: np.ndarray
    boundary: np.ndarray
    coefficient_values: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


def graph_from_system(system: RBFSystem, label: str) -> RBFGraph:
    """Construct graph features from the actual sparse RBF-FD operator."""

    coo = system.matrix.tocoo()
    source = coo.col.astype(np.int64)
    target = coo.row.astype(np.int64)
    relative = system.nodes[source] - system.nodes[target]
    distance = np.linalg.norm(relative, axis=1)
    edge_features = np.column_stack(
        (relative[:, 0], relative[:, 1], distance, coo.data)
    )
    coefficient_values = (
        np.ones(system.matrix.shape[0], dtype=float)
        if system.coefficient is None
        else system.coefficient.values(system.nodes)
    )
    return RBFGraph(
        label=label,
        nodes=system.nodes.copy(),
        boundary=system.boundary.copy(),
        coefficient_values=coefficient_values,
        edge_index=np.row_stack((source, target)),
        edge_features=edge_features,
    )


def permute_graph(graph: RBFGraph, permutation: np.ndarray, label: str) -> RBFGraph:
    """Apply ``new = old[permutation]`` consistently to a graph."""

    permutation = np.asarray(permutation, dtype=int)
    n_nodes = graph.nodes.shape[0]
    if permutation.shape != (n_nodes,) or not np.array_equal(
        np.sort(permutation), np.arange(n_nodes)
    ):
        raise ValueError("permutation must contain every node index exactly once")
    old_to_new = np.argsort(permutation)
    return RBFGraph(
        label=label,
        nodes=graph.nodes[permutation],
        boundary=graph.boundary[permutation],
        coefficient_values=graph.coefficient_values[permutation],
        edge_index=old_to_new[graph.edge_index],
        edge_features=graph.edge_features.copy(),
    )


@dataclass(frozen=True)
class GraphFeatureNormalizer:
    """Training-only feature statistics applied to every graph split."""

    coordinate_mean: Tuple[float, float]
    coordinate_scale: Tuple[float, float]
    coefficient_mean: float
    coefficient_scale: float
    edge_mean: Tuple[float, float, float, float]
    edge_scale: Tuple[float, float, float, float]


def fit_graph_feature_normalizer(
    graphs: Sequence[RBFGraph],
) -> GraphFeatureNormalizer:
    """Fit graph feature scales using training systems only."""

    if not graphs:
        raise ValueError("training graphs are required")
    coordinates = np.vstack([graph.nodes for graph in graphs])
    coefficients = np.concatenate(
        [graph.coefficient_values for graph in graphs]
    )
    edges = np.vstack([graph.edge_features for graph in graphs])
    coordinate_scale = np.std(coordinates, axis=0)
    edge_scale = np.std(edges, axis=0)
    coefficient_scale = float(np.std(coefficients))
    coordinate_scale[coordinate_scale <= np.finfo(float).eps] = 1.0
    edge_scale[edge_scale <= np.finfo(float).eps] = 1.0
    if coefficient_scale <= np.finfo(float).eps:
        coefficient_scale = 1.0
    return GraphFeatureNormalizer(
        coordinate_mean=tuple(np.mean(coordinates, axis=0)),
        coordinate_scale=tuple(coordinate_scale),
        coefficient_mean=float(np.mean(coefficients)),
        coefficient_scale=coefficient_scale,
        edge_mean=tuple(np.mean(edges, axis=0)),
        edge_scale=tuple(edge_scale),
    )


@dataclass(frozen=True)
class DirectGNNConfig:
    """Small, deliberately shallow message-passing architecture and trainer."""

    hidden_width: int = 32
    message_layers: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 16
    max_epochs: int = 180
    patience: int = 25
    seed: int = 20260829


class MessageLayer(nn.Module):
    """One edge-message/mean-aggregation/update layer."""

    def __init__(self, width: int, edge_width: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * width + edge_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index[0], edge_index[1]
        batch = hidden.shape[0]
        edge_values = edge_features.unsqueeze(0).expand(batch, -1, -1)
        messages = self.message(
            torch.cat((hidden[:, target], hidden[:, source], edge_values), dim=-1)
        )
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(1, target, messages)
        aggregate = aggregate / degree.view(1, -1, 1)
        return self.norm(hidden + self.update(torch.cat((hidden, aggregate), dim=-1)))


class DirectComplementGNN(nn.Module):
    """Permutation-equivariant nodewise missing-correction predictor."""

    def __init__(
        self,
        node_feature_width: int,
        edge_feature_width: int,
        config: DirectGNNConfig,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_width, config.hidden_width),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                MessageLayer(config.hidden_width, edge_feature_width)
                for _ in range(config.message_layers)
            ]
        )
        self.output = nn.Linear(config.hidden_width, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(node_features)
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_features, degree)
        return self.output(hidden).squeeze(-1)


@dataclass
class _PreparedGraph:
    static_node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    degree: torch.Tensor


def _prepare_graph(
    graph: RBFGraph,
    normalizer: GraphFeatureNormalizer,
) -> _PreparedGraph:
    coordinate_mean = np.asarray(normalizer.coordinate_mean)
    coordinate_scale = np.asarray(normalizer.coordinate_scale)
    normalized_coordinates = (graph.nodes - coordinate_mean) / coordinate_scale
    normalized_coefficient = (
        graph.coefficient_values - normalizer.coefficient_mean
    ) / normalizer.coefficient_scale
    static = np.column_stack(
        (
            normalized_coordinates,
            graph.boundary.astype(float),
            normalized_coefficient,
        )
    )
    edge_mean = np.asarray(normalizer.edge_mean)
    edge_scale = np.asarray(normalizer.edge_scale)
    edges = (graph.edge_features - edge_mean) / edge_scale
    target = graph.edge_index[1]
    degree = np.bincount(target, minlength=graph.nodes.shape[0]).astype(float)
    degree[degree < 1.0] = 1.0
    return _PreparedGraph(
        static_node_features=torch.from_numpy(static.astype(np.float32)),
        edge_index=torch.from_numpy(graph.edge_index.astype(np.int64)),
        edge_features=torch.from_numpy(edges.astype(np.float32)),
        degree=torch.from_numpy(degree.astype(np.float32)),
    )


def _node_features(
    prepared: _PreparedGraph,
    normalized_vectors: np.ndarray,
    normalized_system_features: np.ndarray,
) -> torch.Tensor:
    directions = torch.from_numpy(normalized_vectors.astype(np.float32))
    systems = torch.from_numpy(normalized_system_features.astype(np.float32))
    batch, n_nodes = directions.shape
    static = prepared.static_node_features.unsqueeze(0).expand(batch, -1, -1)
    repeated_systems = systems.unsqueeze(1).expand(-1, n_nodes, -1)
    return torch.cat((static, directions.unsqueeze(-1), repeated_systems), dim=-1)


@dataclass
class GNNTrainingResult:
    predictor: "DirectGNNPredictor"
    history: List[Dict[str, float]]
    training_seconds: float
    best_epoch: int
    best_validation_loss: float


class DirectGNNPredictor:
    """NumPy-facing, positively homogeneous graph predictor."""

    def __init__(
        self,
        model: DirectComplementGNN,
        correction_normalizer: DirectNormalizer,
        graph_normalizer: GraphFeatureNormalizer,
        graphs: Mapping[str, RBFGraph],
    ) -> None:
        self.model = model.eval()
        self.correction_normalizer = correction_normalizer
        self.graph_normalizer = graph_normalizer
        self.graphs: Dict[str, RBFGraph] = dict(graphs)
        self.prepared = {
            label: _prepare_graph(graph, graph_normalizer)
            for label, graph in self.graphs.items()
        }
        self.inference_seconds = 0.0
        self.inference_count = 0

    def register_graph(self, graph: RBFGraph) -> None:
        self.graphs[graph.label] = graph
        self.prepared[graph.label] = _prepare_graph(graph, self.graph_normalizer)

    def reset_timing(self) -> None:
        self.inference_seconds = 0.0
        self.inference_count = 0

    def predict_batch(
        self,
        vectors: np.ndarray,
        system_features: np.ndarray,
        graph_label: str,
    ) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=float)
        system_features = np.asarray(system_features, dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("zero vector cannot be predicted")
        directions = vectors / norms[:, None]
        mean = np.asarray(self.correction_normalizer.system_mean)
        scale = np.asarray(self.correction_normalizer.system_scale)
        normalized_systems = (system_features - mean) / scale
        graph = self.prepared[graph_label]
        start = perf_counter()
        node_features = _node_features(graph, directions, normalized_systems)
        with torch.no_grad():
            encoded = self.model(
                node_features,
                graph.edge_index,
                graph.edge_features,
                graph.degree,
            ).cpu().numpy().astype(float)
        prediction = self.correction_normalizer.decode_targets(encoded, norms)
        self.inference_seconds += perf_counter() - start
        self.inference_count += vectors.shape[0]
        return prediction

    def predict(
        self,
        vector: np.ndarray,
        system_features: np.ndarray,
        graph_label: str,
    ) -> np.ndarray:
        return self.predict_batch(
            np.asarray(vector)[None, :],
            np.asarray(system_features)[None, :],
            graph_label,
        )[0]

    def save(
        self,
        path: Path,
        config: DirectGNNConfig,
        history: Sequence[Dict[str, float]],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "correction_normalizer": asdict(self.correction_normalizer),
                "graph_normalizer": asdict(self.graph_normalizer),
                "config": asdict(config),
                "history": list(history),
            },
            path,
        )


def train_direct_gnn(
    train_vectors: np.ndarray,
    train_corrections: np.ndarray,
    train_system_features: np.ndarray,
    train_system_labels: np.ndarray,
    validation_vectors: np.ndarray,
    validation_corrections: np.ndarray,
    validation_system_features: np.ndarray,
    validation_system_labels: np.ndarray,
    graphs: Mapping[str, RBFGraph],
    correction_normalizer: DirectNormalizer,
    config: DirectGNNConfig,
) -> GNNTrainingResult:
    """Train the GNN with the same normalized target used by the MLP."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    train_graph_labels = sorted(set(train_system_labels.tolist()))
    graph_normalizer = fit_graph_feature_normalizer(
        [graphs[label] for label in train_graph_labels]
    )
    prepared = {
        label: _prepare_graph(graph, graph_normalizer)
        for label, graph in graphs.items()
    }
    train_norms = np.linalg.norm(train_vectors, axis=1)
    validation_norms = np.linalg.norm(validation_vectors, axis=1)
    train_directions = train_vectors / train_norms[:, None]
    validation_directions = validation_vectors / validation_norms[:, None]
    target_train = correction_normalizer.encode_targets(
        train_corrections, train_norms
    ).astype(np.float32)
    target_validation = correction_normalizer.encode_targets(
        validation_corrections, validation_norms
    ).astype(np.float32)
    system_mean = np.asarray(correction_normalizer.system_mean)
    system_scale = np.asarray(correction_normalizer.system_scale)
    train_systems = (train_system_features - system_mean) / system_scale
    validation_systems = (
        validation_system_features - system_mean
    ) / system_scale
    node_feature_width = 5 + train_system_features.shape[1]
    model = DirectComplementGNN(
        node_feature_width,
        next(iter(prepared.values())).edge_features.shape[1],
        config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    rng = np.random.default_rng(config.seed)

    def batch_loss(
        indices: np.ndarray,
        directions: np.ndarray,
        systems: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray,
    ) -> torch.Tensor:
        total = None
        count = 0
        for label in sorted(set(labels[indices].tolist())):
            local = indices[labels[indices] == label]
            graph = prepared[label]
            inputs = _node_features(graph, directions[local], systems[local])
            prediction = model(
                inputs,
                graph.edge_index,
                graph.edge_features,
                graph.degree,
            )
            target = torch.from_numpy(targets[local])
            group_loss = loss_function(prediction, target)
            weighted = group_loss * int(local.size)
            total = weighted if total is None else total + weighted
            count += int(local.size)
        if total is None:
            raise RuntimeError("empty graph batch")
        return total / count

    history: List[Dict[str, float]] = []
    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = rng.permutation(train_vectors.shape[0])
        losses = []
        for offset in range(0, order.size, config.batch_size):
            indices = order[offset : offset + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(
                indices,
                train_directions,
                train_systems,
                train_system_labels,
                target_train,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            all_indices = np.arange(validation_vectors.shape[0])
            for offset in range(0, all_indices.size, config.batch_size):
                indices = all_indices[offset : offset + config.batch_size]
                validation_losses.append(
                    float(
                        batch_loss(
                            indices,
                            validation_directions,
                            validation_systems,
                            validation_system_labels,
                            target_validation,
                        )
                    )
                )
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if np.isfinite(validation_loss) and validation_loss < best_validation - 1.0e-7:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    training_seconds = perf_counter() - start
    if best_state is None:
        raise RuntimeError("GNN training did not produce a finite validation state")
    model.load_state_dict(best_state)
    predictor = DirectGNNPredictor(
        model,
        correction_normalizer,
        graph_normalizer,
        graphs,
    )
    return GNNTrainingResult(
        predictor=predictor,
        history=history,
        training_seconds=training_seconds,
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
    )


def make_direct_gnn_preconditioner(
    base: InverseAction,
    predictor: DirectGNNPredictor,
    system_features: np.ndarray,
    graph_label: str,
) -> InverseAction:
    r"""Return ``M^-1 v + ||v|| N_theta(v/||v||, xi, graph)``."""

    features = np.asarray(system_features, dtype=float)

    def action(vector: np.ndarray) -> np.ndarray:
        return base.apply(vector) + predictor.predict(vector, features, graph_label)

    return InverseAction(
        f"{base.name}+direct_gnn",
        action,
        setup_seconds=base.setup_seconds,
        is_linear=False,
        diagnostics={"learned_strategy": "direct_missing_action_graph"},
    )
