"""Stage 14 diagnostics for subspace-learning targets and solver-aware losses.

The experiment keeps the Stage 10--13 RBF-FD systems, strong ILU, archived
right Ritz spaces, and fixed-space right-preconditioned recycled GMRES.  It
changes only the candidate right space supplied to that solver.  No neural
model is trained.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as la
from scipy.stats import spearmanr

from global_rbf.transfer import grassmann_geodesic
from neurkitt_oracle.gcrodr import restarted_projected_gmres

from ..coarse_space import AZPetrovFactors, factor_az_petrov
from ..preconditioners import PreconditionerFailure
from ..subspace_analysis import ritz_atomic_blocks, subspace_metrics
from .stage9_scalability import _write_csv, _write_json
from .stage10_ritz_transfer import (
    BLUE,
    CHARCOAL,
    LIGHT_GRAY,
    OLIVE,
    ORANGE,
    PINK,
    PURPLE,
    RitzContext,
    Stage10Settings,
    _discrete_rhs,
)
from .stage13_persistent_vs_recycling import (
    Stage13Settings,
    _run_baseline,
    _run_recycled,
    build_fixed_contexts,
)


@dataclass(frozen=True)
class Stage14Settings:
    """Frozen candidate library, solver controls, and decision thresholds."""

    stage10: Stage10Settings = Stage10Settings()
    target_ratios: Tuple[float, ...] = (90.0, 95.0, 100.0, 105.0, 110.0)
    q: int = 8
    random_seed: int = 20261401
    basis_rotation_replicates: int = 2
    random_perturbation_replicates: int = 2
    random_perturbation_angles_degrees: Tuple[float, ...] = (5.0, 15.0, 30.0, 60.0)
    interpolation_fractions: Tuple[float, ...] = (0.25, 0.50, 0.75)
    short_horizons: Tuple[int, ...] = (1, 2, 5, 10)
    correlation_minimum_n: int = 8
    reliable_overall_spearman: float = 0.55
    reliable_target_median_spearman: float = 0.45
    reliable_family_spearman: float = 0.30
    reliable_family_count: int = 3
    reliable_lofo_skill: float = 0.05
    reliable_lofo_spearman: float = 0.45
    solver_aware_margin: float = 0.10
    lofo_margin: float = 0.05
    go_max_short_horizon: int = 5

    @property
    def stage13(self) -> Stage13Settings:
        return Stage13Settings(
            stage10=self.stage10,
            ratios=self.target_ratios,
            q=self.q,
            transfer_pairs=tuple(
                (self.target_ratios[index], self.target_ratios[index + 1])
                for index in range(len(self.target_ratios) - 1)
            ),
            representative_ratios=self.target_ratios,
        )


@dataclass(frozen=True)
class CandidateSpace:
    """One deterministic candidate right space and its provenance."""

    candidate_id: str
    target_ratio: float
    family: str
    subtype: str
    basis: np.ndarray
    source_ratio: float | str = ""
    second_source_ratio: float | str = ""
    interpolation_fraction: float | str = ""
    perturbation_angle_degrees: float | str = ""
    replicate: int | str = ""
    block_start: int | str = ""
    block_stop: int | str = ""


METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("d_z", r"$d_Z$", "identity"),
    ("d_w", r"$d_W$", "identity"),
    ("short_rho_1_median", r"$\rho_1$", "log10"),
    ("short_rho_2_median", r"$\rho_2$", "log10"),
    ("short_rho_5_median", r"$\rho_5$", "log10"),
    ("short_rho_10_median", r"$\rho_{10}$", "log10"),
    ("weighted_geometry", r"$d_\omega$", "identity"),
)

ANALYSIS_FAMILIES: Tuple[str, ...] = (
    "reuse",
    "interpolation",
    "random_perturbation",
    "partial_block_replacement",
)


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Return a full-rank economic QR basis with deterministic column signs."""

    array = np.asarray(basis, dtype=float)
    if array.ndim != 2 or array.shape[0] < array.shape[1]:
        raise ValueError("basis must have at least as many rows as columns")
    orthonormal, triangular = la.qr(array, mode="economic", check_finite=False)
    diagonal = np.diag(triangular)
    scale = max(float(np.linalg.norm(triangular, ord=2)), 1.0)
    if np.min(np.abs(diagonal)) <= 100.0 * np.finfo(float).eps * scale:
        raise ValueError("candidate basis is rank deficient")
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return orthonormal * signs


def _random_complement(
    oracle: np.ndarray,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate ``width`` orthonormal vectors perpendicular to ``oracle``."""

    if width < 1 or width > oracle.shape[1]:
        raise ValueError("invalid complement width")
    for _ in range(8):
        raw = rng.normal(size=(oracle.shape[0], width))
        for _ in range(2):
            raw -= oracle @ (oracle.T @ raw)
        try:
            complement = _orthonormalize(raw)
        except ValueError:
            continue
        if np.linalg.norm(oracle.T @ complement, ord="fro") <= 1.0e-10:
            return complement
    raise RuntimeError("failed to generate an oracle-orthogonal complement")


def controlled_perturbation(
    oracle: np.ndarray,
    angle_degrees: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Move every principal direction by the same controlled Grassmann angle."""

    angle = np.radians(float(angle_degrees))
    complement = _random_complement(oracle, oracle.shape[1], rng)
    return _orthonormalize(np.cos(angle) * oracle + np.sin(angle) * complement)


def replace_atomic_block(
    oracle: np.ndarray,
    start: int,
    stop: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace one real/complex atomic Ritz block by orthogonal random directions."""

    if not 0 <= start < stop <= oracle.shape[1]:
        raise ValueError("invalid block interval")
    replacement = _random_complement(oracle, stop - start, rng)
    columns = []
    for index in range(oracle.shape[1]):
        if index == start:
            columns.extend(replacement[:, offset] for offset in range(stop - start))
        if start <= index < stop:
            continue
        columns.append(oracle[:, index])
    return _orthonormalize(np.column_stack(columns))


def weighted_block_geometry(
    candidate: np.ndarray,
    oracle: np.ndarray,
    blocks: Sequence[Tuple[int, int]],
    weights: Sequence[float],
) -> float:
    r"""Return importance-weighted missed oracle energy in ``[0,1]``.

    For block ``b``, the missed energy is
    ``||(I-P_candidate) Z_b||_F^2``.  Block penalties are normalized only
    through the denominator, so zero-importance blocks do not affect the
    metric.  This is basis invariant within every atomic block.
    """

    if len(blocks) != len(weights):
        raise ValueError("blocks and weights must have equal length")
    numerator = 0.0
    denominator = 0.0
    for (start, stop), weight in zip(blocks, weights):
        positive = max(float(weight), 0.0)
        if positive == 0.0:
            continue
        block = oracle[:, start:stop]
        missed = block - candidate @ (candidate.T @ block)
        numerator += positive * float(np.linalg.norm(missed, ord="fro") ** 2)
        denominator += positive * (stop - start)
    return float(np.sqrt(numerator / denominator)) if denominator > 0.0 else float("nan")


def _basis_hash(basis: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(basis, dtype=np.float64).tobytes()).hexdigest()


def _candidate_id(target: float, family: str, label: str) -> str:
    safe = label.replace(".", "p").replace("-", "m").replace(" ", "_")
    return f"R{target:g}_{family}_{safe}"


def generate_candidates(
    target_ratio: float,
    spaces: Mapping[float, np.ndarray],
    eigenvalues: Mapping[float, np.ndarray],
    settings: Stage14Settings,
) -> Tuple[List[CandidateSpace], Tuple[Tuple[int, int], ...]]:
    """Build a balanced, deterministic library around one frozen oracle space."""

    oracle = spaces[target_ratio]
    ratios = tuple(sorted(spaces))
    target_index = ratios.index(target_ratio)
    rng = np.random.default_rng(settings.random_seed + int(100 * target_ratio))
    blocks = ritz_atomic_blocks(eigenvalues[target_ratio][: settings.q])
    candidates: List[CandidateSpace] = []

    def add(family: str, subtype: str, label: str, basis: np.ndarray, **metadata: Any) -> None:
        candidates.append(
            CandidateSpace(
                candidate_id=_candidate_id(target_ratio, family, label),
                target_ratio=target_ratio,
                family=family,
                subtype=subtype,
                basis=_orthonormalize(basis),
                **metadata,
            )
        )

    add("oracle", "oracle", "own", oracle, source_ratio=target_ratio)

    for replicate in range(settings.basis_rotation_replicates):
        rotation = _orthonormalize(rng.normal(size=(settings.q, settings.q)))
        add(
            "basis_rotation",
            "within_range_rotation",
            f"rep{replicate + 1}",
            oracle @ rotation,
            source_ratio=target_ratio,
            replicate=replicate + 1,
        )

    for donor_ratio in ratios:
        if donor_ratio == target_ratio:
            continue
        add(
            "reuse",
            "near" if abs(donor_ratio - target_ratio) <= 5.0 else "far",
            f"donor{donor_ratio:g}",
            spaces[donor_ratio],
            source_ratio=donor_ratio,
        )

    neighbor_indices = [index for index in (target_index - 1, target_index + 1) if 0 <= index < len(ratios)]
    for neighbor_index in neighbor_indices:
        donor_ratio = ratios[neighbor_index]
        for fraction in settings.interpolation_fractions:
            add(
                "interpolation",
                "oracle_to_neighbor_geodesic",
                f"own_to_{donor_ratio:g}_t{fraction:.2f}",
                grassmann_geodesic(oracle, spaces[donor_ratio], fraction),
                source_ratio=target_ratio,
                second_source_ratio=donor_ratio,
                interpolation_fraction=fraction,
            )

    if 0 < target_index < len(ratios) - 1:
        left_ratio = ratios[target_index - 1]
        right_ratio = ratios[target_index + 1]
        for fraction in settings.interpolation_fractions:
            add(
                "interpolation",
                "bracketing_geodesic",
                f"bridge_{left_ratio:g}_{right_ratio:g}_t{fraction:.2f}",
                grassmann_geodesic(spaces[left_ratio], spaces[right_ratio], fraction),
                source_ratio=left_ratio,
                second_source_ratio=right_ratio,
                interpolation_fraction=fraction,
            )

    for replicate in range(settings.random_perturbation_replicates):
        for angle in settings.random_perturbation_angles_degrees:
            add(
                "random_perturbation",
                "uniform_principal_angle",
                f"deg{angle:g}_rep{replicate + 1}",
                controlled_perturbation(oracle, angle, rng),
                source_ratio=target_ratio,
                perturbation_angle_degrees=angle,
                replicate=replicate + 1,
            )

    for block_index, (start, stop) in enumerate(blocks):
        add(
            "partial_block_replacement",
            "atomic_ritz_block",
            f"block{block_index + 1}_{start}_{stop}",
            replace_atomic_block(oracle, start, stop, rng),
            source_ratio=target_ratio,
            replicate=block_index + 1,
            block_start=start,
            block_stop=stop,
        )

    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("candidate identifiers are not unique")
    return candidates, blocks


def _load_eigenvalues(output: Path, ratios: Iterable[float]) -> Dict[float, np.ndarray]:
    values: Dict[float, np.ndarray] = {}
    for ratio in ratios:
        path = output / "spaces" / f"fixed_stage10_ritz_R{ratio:g}_q32.npz"
        with np.load(path) as archive:
            values[ratio] = np.asarray(archive["eigenvalues"], dtype=complex)
    return values


def _short_rollout(
    context: RitzContext,
    rhs: np.ndarray,
    factors: AZPetrovFactors,
    settings: Stage14Settings,
) -> Tuple[Dict[int, float], float, Dict[str, float]]:
    """Run a separate capped projected solve and return true residual ratios."""

    context.ilu.reset_timing()
    start = perf_counter()
    result = restarted_projected_gmres(
        context.system.matrix,
        rhs,
        context.ilu.apply,
        u_basis=factors.recycling_trial,
        c_basis=factors.test,
        rtol=settings.stage10.solver.relative_tolerance,
        atol=settings.stage10.solver.absolute_tolerance,
        restart=settings.stage10.solver.restart,
        max_iterations=max(settings.short_horizons),
        record_true_residuals=True,
        record_projection_diagnostics=False,
    )
    elapsed = perf_counter() - start
    history = {
        int(iteration): float(value)
        for iteration, value in zip(result.true_residual_iterations, result.true_residual_history)
    }
    ratios: Dict[int, float] = {}
    for horizon in settings.short_horizons:
        available = [iteration for iteration in history if iteration <= horizon]
        if not available:
            ratios[horizon] = float("nan")
        else:
            ratios[horizon] = history[max(available)] / max(history[0], np.finfo(float).tiny)
    diagnostics = result.diagnostics or {}
    return ratios, elapsed, {key: float(value) for key, value in diagnostics.items()}


def _aggregate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    settings: Stage14Settings,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((float(row["target_ratio"]), str(row["candidate_id"])), []).append(row)
    summaries: List[Dict[str, Any]] = []
    for (ratio, candidate_id), local in groups.items():
        first = local[0]
        result = {
            key: first[key]
            for key in (
                "target_ratio",
                "candidate_id",
                "family",
                "subtype",
                "source_ratio",
                "second_source_ratio",
                "interpolation_fraction",
                "perturbation_angle_degrees",
                "replicate",
                "block_start",
                "block_stop",
                "basis_sha256",
                "d_z",
                "d_z_normalized",
                "d_w",
                "d_w_normalized",
                "principal_angle_max_degrees",
                "w_principal_angle_max_degrees",
                "condition_wtaz",
                "setup_seconds",
                "setup_failure",
            )
        }
        result["failure"] = first["setup_failure"]
        valid = [row for row in local if row["setup_failure"] == ""]
        result["rhs_count"] = len(local)
        result["valid_rhs_count"] = len(valid)
        if valid:
            for field in ("iterations", "j_iter", "online_solve_seconds", "short_rollout_seconds"):
                result[f"{field}_median"] = float(np.median([float(row[field]) for row in valid]))
            for horizon in settings.short_horizons:
                field = f"short_rho_{horizon}"
                result[f"{field}_median"] = float(np.median([float(row[field]) for row in valid]))
            result["convergence_rate"] = float(np.mean([bool(row["converged"]) for row in valid]))
            result["true_final_relative_residual_max"] = float(
                max(float(row["true_final_relative_residual"]) for row in valid)
            )
        else:
            for field in ("iterations_median", "j_iter_median", "online_solve_seconds_median", "short_rollout_seconds_median"):
                result[field] = float("nan")
            for horizon in settings.short_horizons:
                result[f"short_rho_{horizon}_median"] = float("nan")
            result["convergence_rate"] = 0.0
            result["true_final_relative_residual_max"] = float("nan")
        summaries.append(result)
    return summaries


def run_baselines(
    output: Path,
    contexts: Mapping[float, RitzContext],
    factors: Mapping[float, AZPetrovFactors],
    settings: Stage14Settings,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[float, int], int]]:
    """Run the unchanged strong-ILU baseline once per target and held-out RHS."""

    rows: List[Dict[str, Any]] = []
    lookup: Dict[Tuple[float, int], int] = {}
    for ratio in settings.target_ratios:
        for rhs_seed in settings.stage10.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(contexts[ratio].system, rhs_seed)
            result, counters, online = _run_baseline(
                contexts[ratio], rhs, factors[ratio], settings.stage13
            )
            row = {
                "target_ratio": ratio,
                "rhs_seed": rhs_seed,
                "iterations": result.iterations,
                "converged": result.converged,
                "cap_hit": bool(result.iterations >= settings.stage10.solver.max_iterations and not result.converged),
                "true_final_relative_residual": float(result.relative_residuals[-1]),
                "online_solve_seconds": online,
                "a_matvecs": counters.a_matvecs,
                "ilu_solves": counters.ilu_solves,
                "failure": result.failure,
            }
            rows.append(row)
            lookup[(ratio, rhs_seed)] = int(result.iterations)
    _write_csv(output / "baseline_ilu_results.csv", rows)
    return rows, lookup


def run_candidate_library(
    output: Path,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    eigenvalues: Mapping[float, np.ndarray],
    baseline_iterations: Mapping[Tuple[float, int], int],
    settings: Stage14Settings,
) -> Tuple[
    Dict[float, List[CandidateSpace]],
    Dict[float, Tuple[Tuple[int, int], ...]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Evaluate all candidate spaces with separate short and full recycled solves."""

    candidate_map: Dict[float, List[CandidateSpace]] = {}
    block_map: Dict[float, Tuple[Tuple[int, int], ...]] = {}
    rows: List[Dict[str, Any]] = []
    basis_archive: Dict[str, np.ndarray] = {}
    index_rows: List[Dict[str, Any]] = []
    archive_index = 0
    for ratio in settings.target_ratios:
        candidates, blocks = generate_candidates(ratio, spaces, eigenvalues, settings)
        candidate_map[ratio] = candidates
        block_map[ratio] = blocks
        oracle = spaces[ratio]
        oracle_pair = factor_az_petrov(contexts[ratio].system.matrix, oracle)
        for candidate in candidates:
            archive_key = f"basis_{archive_index:04d}"
            archive_index += 1
            basis_archive[archive_key] = candidate.basis
            basis_hash = _basis_hash(candidate.basis)
            failure = ""
            try:
                setup_start = perf_counter()
                pair = factor_az_petrov(contexts[ratio].system.matrix, candidate.basis)
                setup_seconds = perf_counter() - setup_start
                z_metrics = subspace_metrics(candidate.basis, oracle)
                w_metrics = subspace_metrics(pair.test, oracle_pair.test)
                geometry = {
                    "d_z": float(z_metrics["projector_distance"]),
                    "d_z_normalized": float(z_metrics["projector_distance_normalized"]),
                    "d_w": float(w_metrics["projector_distance"]),
                    "d_w_normalized": float(w_metrics["projector_distance_normalized"]),
                    "principal_angle_max_degrees": float(z_metrics["principal_angle_max_degrees"]),
                    "w_principal_angle_max_degrees": float(w_metrics["principal_angle_max_degrees"]),
                    "condition_wtaz": float(pair.diagnostics["wtaz_condition"]),
                }
            except (PreconditionerFailure, ValueError, la.LinAlgError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                pair = None
                setup_seconds = float("nan")
                geometry = {
                    "d_z": float("nan"),
                    "d_z_normalized": float("nan"),
                    "d_w": float("nan"),
                    "d_w_normalized": float("nan"),
                    "principal_angle_max_degrees": float("nan"),
                    "w_principal_angle_max_degrees": float("nan"),
                    "condition_wtaz": float("nan"),
                }
            metadata = {
                "target_ratio": ratio,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "subtype": candidate.subtype,
                "source_ratio": candidate.source_ratio,
                "second_source_ratio": candidate.second_source_ratio,
                "interpolation_fraction": candidate.interpolation_fraction,
                "perturbation_angle_degrees": candidate.perturbation_angle_degrees,
                "replicate": candidate.replicate,
                "block_start": candidate.block_start,
                "block_stop": candidate.block_stop,
                "basis_sha256": basis_hash,
                "basis_archive_key": archive_key,
                "q": settings.q,
                "setup_seconds": setup_seconds,
                "setup_failure": failure,
                "failure": failure,
                **geometry,
            }
            index_rows.append(metadata)
            for rhs_seed in settings.stage10.evaluation_rhs_seeds:
                row = dict(metadata)
                row["rhs_seed"] = rhs_seed
                if pair is None:
                    row.update(
                        {
                            "iterations": settings.stage10.solver.max_iterations,
                            "converged": False,
                            "cap_hit": True,
                            "true_final_relative_residual": float("nan"),
                            "online_solve_seconds": float("nan"),
                            "short_rollout_seconds": float("nan"),
                            "j_iter": float("nan"),
                        }
                    )
                    for horizon in settings.short_horizons:
                        row[f"short_rho_{horizon}"] = float("nan")
                    rows.append(row)
                    continue
                rhs, _ = _discrete_rhs(contexts[ratio].system, rhs_seed)
                short_ratios, short_seconds, short_diagnostics = _short_rollout(
                    contexts[ratio], rhs, pair, settings
                )
                result, counters, online = _run_recycled(
                    contexts[ratio], rhs, pair, settings.stage13
                )
                baseline = baseline_iterations[(ratio, rhs_seed)]
                row.update(
                    {
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "cap_hit": bool(result.iterations >= settings.stage10.solver.max_iterations and not result.converged),
                        "true_final_relative_residual": result.relative_residual,
                        "online_solve_seconds": online,
                        "short_rollout_seconds": short_seconds,
                        "short_rollout_max_iterations": max(settings.short_horizons),
                        "full_solve_max_iterations": settings.stage10.solver.max_iterations,
                        "short_rollout_independent": True,
                        "j_iter": (baseline - result.iterations) / max(baseline, 1),
                        "a_matvecs": int((result.operation_counts or {}).get("a_matvecs", 0)),
                        "ilu_solves": counters.ilu_solves,
                        "short_projected_arnoldi_relation_error": short_diagnostics.get(
                            "projected_arnoldi_relation_error_max", float("nan")
                        ),
                        "failure": "" if result.converged else "maximum iterations reached",
                    }
                )
                for horizon, value in short_ratios.items():
                    row[f"short_rho_{horizon}"] = value
                rows.append(row)
        _write_csv(output / "candidate_results.csv", rows)
        _write_csv(output / "candidate_index.csv", index_rows)

    np.savez_compressed(output / "candidate_spaces.npz", **basis_archive)
    summaries = _aggregate_candidate_rows(rows, settings)
    _write_csv(output / "candidate_summary.csv", summaries)
    return candidate_map, block_map, rows, summaries


def run_block_importance(
    output: Path,
    contexts: Mapping[float, RitzContext],
    spaces: Mapping[float, np.ndarray],
    blocks: Mapping[float, Sequence[Tuple[int, int]]],
    candidate_rows: Sequence[Mapping[str, Any]],
    settings: Stage14Settings,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[float, np.ndarray]]:
    """Estimate atomic-block importance from held-out deletion penalties."""

    oracle_iterations = {
        (float(row["target_ratio"]), int(row["rhs_seed"])): int(row["iterations"])
        for row in candidate_rows
        if row["family"] == "oracle"
    }
    raw: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    weights: Dict[float, np.ndarray] = {}
    for ratio in settings.target_ratios:
        local_summaries: List[Dict[str, Any]] = []
        oracle = spaces[ratio]
        for block_index, (start, stop) in enumerate(blocks[ratio]):
            retained = np.delete(oracle, np.arange(start, stop), axis=1)
            retained = _orthonormalize(retained)
            pair = factor_az_petrov(contexts[ratio].system.matrix, retained)
            penalties = []
            for rhs_seed in settings.stage10.evaluation_rhs_seeds:
                rhs, _ = _discrete_rhs(contexts[ratio].system, rhs_seed)
                result, _, online = _run_recycled(
                    contexts[ratio], rhs, pair, settings.stage13
                )
                penalty = result.iterations - oracle_iterations[(ratio, rhs_seed)]
                penalties.append(penalty)
                raw.append(
                    {
                        "target_ratio": ratio,
                        "block_index": block_index,
                        "block_start": start,
                        "block_stop": stop,
                        "block_size": stop - start,
                        "rhs_seed": rhs_seed,
                        "oracle_iterations": oracle_iterations[(ratio, rhs_seed)],
                        "deleted_iterations": result.iterations,
                        "iteration_penalty": penalty,
                        "converged": result.converged,
                        "online_solve_seconds": online,
                    }
                )
            summary = {
                "target_ratio": ratio,
                "block_index": block_index,
                "block_start": start,
                "block_stop": stop,
                "block_size": stop - start,
                "iteration_penalty_median": float(np.median(penalties)),
                "iteration_penalty_min": int(min(penalties)),
                "iteration_penalty_max": int(max(penalties)),
                "positive_importance": max(float(np.median(penalties)), 0.0),
                "rhs_sign_consistent": bool(min(penalties) >= 0 or max(penalties) <= 0),
            }
            local_summaries.append(summary)
            summaries.append(summary)
        positive = np.asarray([row["positive_importance"] for row in local_summaries], dtype=float)
        weights[ratio] = positive
        denominator = float(np.sum(positive))
        for row, weight in zip(local_summaries, positive):
            row["normalized_weight"] = weight / denominator if denominator > 0.0 else float("nan")
    _write_csv(output / "block_deletion_results.csv", raw)
    _write_csv(output / "block_importance.csv", summaries)
    return raw, summaries, weights


def attach_weighted_geometry(
    candidates: Mapping[float, Sequence[CandidateSpace]],
    blocks: Mapping[float, Sequence[Tuple[int, int]]],
    weights: Mapping[float, np.ndarray],
    spaces: Mapping[float, np.ndarray],
    rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
) -> None:
    """Attach the optional block-weighted geometry to raw and aggregate rows."""

    lookup: Dict[Tuple[float, str], float] = {}
    for ratio, local in candidates.items():
        for candidate in local:
            lookup[(ratio, candidate.candidate_id)] = weighted_block_geometry(
                candidate.basis, spaces[ratio], blocks[ratio], weights[ratio]
            )
    for collection in (rows, summaries):
        for row in collection:
            row["weighted_geometry"] = lookup[
                (float(row["target_ratio"]), str(row["candidate_id"]))
            ]


def _finite_pairs(rows: Sequence[Mapping[str, Any]], x_field: str, y_field: str) -> Tuple[np.ndarray, np.ndarray]:
    pairs = [
        (float(row[x_field]), float(row[y_field]))
        for row in rows
        if np.isfinite(float(row.get(x_field, float("nan"))))
        and np.isfinite(float(row.get(y_field, float("nan"))))
    ]
    if not pairs:
        return np.empty(0), np.empty(0)
    return np.asarray(pairs, dtype=float).T


def _spearman(rows: Sequence[Mapping[str, Any]], x_field: str, y_field: str) -> Dict[str, Any]:
    x, y = _finite_pairs(rows, x_field, y_field)
    if x.size < 3 or np.ptp(x) <= np.finfo(float).eps or np.ptp(y) <= np.finfo(float).eps:
        return {"n": int(x.size), "rho": float("nan"), "p_value": float("nan")}
    result = spearmanr(x, y)
    return {"n": int(x.size), "rho": float(result.statistic), "p_value": float(result.pvalue)}


def compute_correlations(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    settings: Stage14Settings,
) -> List[Dict[str, Any]]:
    """Compute pooled, target-controlled, family, and RHS-level rank correlations."""

    eligible = [row for row in summaries if row["family"] in ANALYSIS_FAMILIES and row["failure"] == ""]
    raw_eligible = [
        row
        for row in raw_rows
        if row["family"] in ANALYSIS_FAMILIES and row["setup_failure"] == ""
    ]
    scopes: List[Tuple[str, Sequence[Mapping[str, Any]]]] = [("overall_nontrivial", eligible)]
    scopes.extend(
        (f"target_R{ratio:g}", [row for row in eligible if float(row["target_ratio"]) == ratio])
        for ratio in settings.target_ratios
    )
    scopes.extend(
        (f"family_{family}", [row for row in eligible if row["family"] == family])
        for family in ANALYSIS_FAMILIES
    )
    scopes.extend(
        (
            f"rhs_{rhs_seed}",
            [row for row in raw_eligible if int(row["rhs_seed"]) == rhs_seed],
        )
        for rhs_seed in settings.stage10.evaluation_rhs_seeds
    )
    rows: List[Dict[str, Any]] = []
    for scope, local in scopes:
        for metric, label, _ in METRICS:
            j_field = "j_iter" if scope.startswith("rhs_") else "j_iter_median"
            iteration_field = "iterations" if scope.startswith("rhs_") else "iterations_median"
            metric_field = (
                metric.removesuffix("_median")
                if scope.startswith("rhs_") and metric.startswith("short_rho_")
                else metric
            )
            iteration = _spearman(local, metric_field, iteration_field)
            benefit = _spearman(local, metric_field, j_field)
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "metric_label": label,
                    "n": benefit["n"],
                    "spearman_vs_iterations": iteration["rho"],
                    "p_vs_iterations": iteration["p_value"],
                    "spearman_vs_j_iter": benefit["rho"],
                    "p_vs_j_iter": benefit["p_value"],
                }
            )
    _write_csv(output / "correlations.csv", rows)
    return rows


def _metric_value(row: Mapping[str, Any], metric: str, transform: str) -> float:
    value = float(row[metric])
    if transform == "log10":
        return float(np.log10(max(value, np.finfo(float).tiny)))
    return value


def leave_one_family_out(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Test cross-family calibration with target fixed effects and one scalar metric."""

    eligible = [row for row in summaries if row["family"] in ANALYSIS_FAMILIES and row["failure"] == ""]
    predictions: List[Dict[str, Any]] = []
    for metric, label, transform in METRICS:
        for held_family in ANALYSIS_FAMILIES:
            train = [row for row in eligible if row["family"] != held_family]
            test = [row for row in eligible if row["family"] == held_family]
            train = [row for row in train if np.isfinite(float(row[metric]))]
            test = [row for row in test if np.isfinite(float(row[metric]))]
            if len(train) < 8 or not test:
                continue
            targets = sorted({float(row["target_ratio"]) for row in eligible})
            scale_values = np.asarray([_metric_value(row, metric, transform) for row in train])
            mean = float(np.mean(scale_values))
            scale = max(float(np.std(scale_values)), np.finfo(float).eps)

            def design(local: Sequence[Mapping[str, Any]], include_metric: bool) -> np.ndarray:
                columns = [np.ones(len(local))]
                columns.extend(
                    np.asarray([float(row["target_ratio"]) == target for row in local], dtype=float)
                    for target in targets[1:]
                )
                if include_metric:
                    columns.append(
                        np.asarray(
                            [(_metric_value(row, metric, transform) - mean) / scale for row in local]
                        )
                    )
                return np.column_stack(columns)

            y_train = np.asarray([float(row["j_iter_median"]) for row in train])
            x_train = design(train, True)
            x_test = design(test, True)
            baseline_train = design(train, False)
            baseline_test = design(test, False)
            coefficients = la.lstsq(x_train, y_train, check_finite=False)[0]
            baseline_coefficients = la.lstsq(baseline_train, y_train, check_finite=False)[0]
            prediction = x_test @ coefficients
            baseline_prediction = baseline_test @ baseline_coefficients
            for row, pred, base_pred in zip(test, prediction, baseline_prediction):
                predictions.append(
                    {
                        "metric": metric,
                        "metric_label": label,
                        "held_family": held_family,
                        "target_ratio": row["target_ratio"],
                        "candidate_id": row["candidate_id"],
                        "actual_j_iter": row["j_iter_median"],
                        "predicted_j_iter": float(pred),
                        "target_only_prediction": float(base_pred),
                    }
                )
    summaries_out: List[Dict[str, Any]] = []
    for metric, label, _ in METRICS:
        local = [row for row in predictions if row["metric"] == metric]
        if not local:
            continue
        actual = np.asarray([float(row["actual_j_iter"]) for row in local])
        predicted = np.asarray([float(row["predicted_j_iter"]) for row in local])
        baseline = np.asarray([float(row["target_only_prediction"]) for row in local])
        mae = float(np.mean(np.abs(predicted - actual)))
        baseline_mae = float(np.mean(np.abs(baseline - actual)))
        rank = spearmanr(predicted, actual)
        summaries_out.append(
            {
                "metric": metric,
                "metric_label": label,
                "n": len(local),
                "lofo_mae": mae,
                "target_only_mae": baseline_mae,
                "lofo_skill_vs_target_only": 1.0 - mae / max(baseline_mae, np.finfo(float).tiny),
                "pooled_prediction_spearman": float(rank.statistic),
                "pooled_prediction_p_value": float(rank.pvalue),
            }
        )
    _write_csv(output / "lofo_predictions.csv", predictions)
    _write_csv(output / "lofo_summary.csv", summaries_out)
    return predictions, summaries_out


def build_metric_scorecard(
    output: Path,
    correlations: Sequence[Mapping[str, Any]],
    lofo: Sequence[Mapping[str, Any]],
    settings: Stage14Settings,
) -> List[Dict[str, Any]]:
    """Summarize reliability gates without choosing a metric after inspection."""

    lookup = {(row["scope"], row["metric"]): row for row in correlations}
    lofo_lookup = {row["metric"]: row for row in lofo}
    scorecard: List[Dict[str, Any]] = []
    for metric, label, _ in METRICS:
        overall = lookup.get(("overall_nontrivial", metric), {})
        target_values = [
            float(lookup[(f"target_R{ratio:g}", metric)]["spearman_vs_j_iter"])
            for ratio in settings.target_ratios
            if np.isfinite(float(lookup[(f"target_R{ratio:g}", metric)]["spearman_vs_j_iter"]))
        ]
        family_values = [
            float(lookup[(f"family_{family}", metric)]["spearman_vs_j_iter"])
            for family in ANALYSIS_FAMILIES
            if np.isfinite(float(lookup[(f"family_{family}", metric)]["spearman_vs_j_iter"]))
        ]
        rhs_values = [
            float(lookup[(f"rhs_{seed}", metric)]["spearman_vs_j_iter"])
            for seed in settings.stage10.evaluation_rhs_seeds
            if np.isfinite(float(lookup[(f"rhs_{seed}", metric)]["spearman_vs_j_iter"]))
        ]
        lofo_row = lofo_lookup.get(metric, {})
        overall_rho = float(overall.get("spearman_vs_j_iter", float("nan")))
        target_median = float(np.median(target_values)) if target_values else float("nan")
        family_consistent = int(sum(value <= -settings.reliable_family_spearman for value in family_values))
        skill = float(lofo_row.get("lofo_skill_vs_target_only", float("nan")))
        prediction_rho = float(lofo_row.get("pooled_prediction_spearman", float("nan")))
        reliable = bool(
            overall_rho <= -settings.reliable_overall_spearman
            and target_median <= -settings.reliable_target_median_spearman
            and family_consistent >= settings.reliable_family_count
            and skill >= settings.reliable_lofo_skill
            and prediction_rho >= settings.reliable_lofo_spearman
        )
        scorecard.append(
            {
                "metric": metric,
                "metric_label": label,
                "overall_spearman_vs_j": overall_rho,
                "overall_spearman_vs_iterations": float(overall.get("spearman_vs_iterations", float("nan"))),
                "target_spearman_median": target_median,
                "target_spearman_min": float(min(target_values)) if target_values else float("nan"),
                "target_spearman_max": float(max(target_values)) if target_values else float("nan"),
                "rhs_spearman_median": float(np.median(rhs_values)) if rhs_values else float("nan"),
                "consistent_family_count": family_consistent,
                "available_family_count": len(family_values),
                "lofo_skill_vs_target_only": skill,
                "lofo_prediction_spearman": prediction_rho,
                "reliable": reliable,
            }
        )
    _write_csv(output / "metric_scorecard.csv", scorecard)
    return scorecard


def make_decision(
    output: Path,
    scorecard: Sequence[Mapping[str, Any]],
    block_summaries: Sequence[Mapping[str, Any]],
    settings: Stage14Settings,
) -> Dict[str, Any]:
    """Apply predeclared target-selection and neural-training gates."""

    lookup = {row["metric"]: row for row in scorecard}
    geometry = [lookup["d_z"], lookup["d_w"]]
    best_geometry = min(
        geometry,
        key=lambda row: float(row["overall_spearman_vs_j"]),
    )
    short = [lookup[f"short_rho_{horizon}_median"] for horizon in settings.short_horizons]
    reliable_short = [
        (horizon, row)
        for horizon, row in zip(settings.short_horizons, short)
        if bool(row["reliable"])
    ]
    shortest_reliable = reliable_short[0] if reliable_short else None
    short_substantially_better = False
    if shortest_reliable is not None:
        _, row = shortest_reliable
        short_substantially_better = bool(
            -float(row["overall_spearman_vs_j"])
            >= -float(best_geometry["overall_spearman_vs_j"]) + settings.solver_aware_margin
            and float(row["lofo_skill_vs_target_only"])
            >= float(best_geometry["lofo_skill_vs_target_only"]) + settings.lofo_margin
        )
    if shortest_reliable is not None and short_substantially_better:
        conclusion = "C"
        horizon = shortest_reliable[0]
        label = f"Short-Krylov solver-aware loss is substantially better; the shortest reliable horizon is m={horizon}."
    elif bool(lookup["d_w"]["reliable"]) and (
        -float(lookup["d_w"]["overall_spearman_vs_j"])
        >= -float(lookup["d_z"]["overall_spearman_vs_j"]) + settings.solver_aware_margin
    ):
        conclusion = "B"
        horizon = None
        label = "Residual/test-space projector loss is better supported than right-space geometry."
    elif bool(lookup["d_z"]["reliable"]):
        conclusion = "A"
        horizon = None
        label = "Direct right-space projector loss is adequate under the reliability gates."
    else:
        conclusion = "D"
        horizon = None
        label = "No tested metric reliably predicts final solver benefit across candidate families."

    # Geometry labels remain unstable across systems according to Stage 10--11.
    # A bounded solver-aware prototype is permitted only when a short horizon
    # passes all cross-family gates at modest unroll cost.
    go = bool(conclusion == "C" and horizon is not None and horizon <= settings.go_max_short_horizon)
    inconsistent_blocks = sum(not bool(row["rhs_sign_consistent"]) for row in block_summaries)
    decision = {
        "conclusion": conclusion,
        "conclusion_label": label,
        "recommended_metric": (
            f"short_rho_{horizon}_median" if conclusion == "C" else
            "d_w" if conclusion == "B" else
            "d_z" if conclusion == "A" else "none"
        ),
        "recommended_short_horizon": horizon,
        "neural_training_decision": "GO" if go else "NO-GO",
        "neural_training_scope": (
            "bounded Stage 15 proof-of-concept predicting an orthonormal right space and optimizing the short recycled-Krylov residual"
            if go else
            "do not train a subspace network until a solver-aligned target passes the cross-family and cost gates"
        ),
        "best_geometry_metric": best_geometry["metric"],
        "best_geometry_overall_spearman": float(best_geometry["overall_spearman_vs_j"]),
        "short_substantially_better": short_substantially_better,
        "block_count": len(block_summaries),
        "block_rhs_sign_inconsistent_count": inconsistent_blocks,
        "thresholds": {
            key: value
            for key, value in asdict(settings).items()
            if key.startswith("reliable_") or key in ("solver_aware_margin", "lofo_margin", "go_max_short_horizon")
        },
    }
    _write_json(output / "decision.json", decision)
    return decision


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_plots(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    scorecard: Sequence[Mapping[str, Any]],
    settings: Stage14Settings,
) -> List[str]:
    """Create relationship, cross-family, horizon, and reuse diagnostics."""

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in summaries if row["family"] in ANALYSIS_FAMILIES and row["failure"] == ""]
    palette = {
        "reuse": BLUE,
        "interpolation": ORANGE,
        "random_perturbation": PURPLE,
        "partial_block_replacement": OLIVE,
    }
    labels = {
        "reuse": "system reuse",
        "interpolation": "Grassmann interpolation",
        "random_perturbation": "random perturbation",
        "partial_block_replacement": "Ritz-block replacement",
    }
    names: List[str] = []

    scatter_metrics = METRICS[:6]
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), sharey=True)
    overall = {(row["metric"]): row for row in correlations if row["scope"] == "overall_nontrivial"}
    for axis, (metric, label, _) in zip(axes.flat, scatter_metrics):
        for family in ANALYSIS_FAMILIES:
            local = [row for row in eligible if row["family"] == family]
            x = np.asarray([float(row[metric]) for row in local])
            if metric.startswith("short_rho"):
                x = np.log10(np.maximum(x, np.finfo(float).tiny))
                x_label = rf"$\log_{{10}}$ {label}"
            else:
                x_label = label
            axis.scatter(
                x,
                [float(row["j_iter_median"]) for row in local],
                s=26,
                alpha=0.72,
                color=palette[family],
                edgecolors="none",
                label=labels[family],
            )
        rho = float(overall[metric]["spearman_vs_j_iter"])
        axis.text(0.03, 0.05, rf"Spearman $\rho={rho:.2f}$", transform=axis.transAxes)
        axis.axhline(0.0, color=LIGHT_GRAY, linestyle="--", linewidth=1.0)
        axis.set(xlabel=x_label, ylabel=r"median $J_{iter}$")
        axis.grid(alpha=0.20)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Candidate-space metrics versus final solver benefit",
        y=1.075,
        x=0.02,
        ha="left",
    )
    name = "metric_vs_solver_benefit.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    lookup = {(row["scope"], row["metric"]): row for row in correlations}
    metric_names = [item[0] for item in METRICS]
    matrix = np.asarray(
        [
            [float(lookup[(f"family_{family}", metric)]["spearman_vs_j_iter"]) for metric in metric_names]
            for family in ANALYSIS_FAMILIES
        ]
    )
    figure, axis = plt.subplots(figsize=(10.2, 4.5))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(column_index, row_index, "NA" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", color="white" if np.isfinite(value) and abs(value) > 0.55 else CHARCOAL)
    axis.set_xticks(range(len(metric_names)), [item[1] for item in METRICS])
    axis.set_yticks(range(len(ANALYSIS_FAMILIES)), [labels[family] for family in ANALYSIS_FAMILIES])
    axis.set_title(r"Within-family Spearman correlation with $J_{iter}$", loc="left")
    figure.colorbar(image, ax=axis, label="Spearman correlation")
    name = "family_correlation_heatmap.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    score_lookup = {row["metric"]: row for row in scorecard}
    horizons = list(settings.short_horizons)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    axes[0].plot(
        horizons,
        [-float(score_lookup[f"short_rho_{horizon}_median"]["overall_spearman_vs_j"]) for horizon in horizons],
        marker="o",
        color=BLUE,
    )
    axes[0].axhline(settings.reliable_overall_spearman, color=CHARCOAL, linestyle="--", linewidth=1.2, label="reliability gate")
    axes[0].set(xlabel="short rollout horizon m", ylabel=r"$-\rho_S(\rho_m,J_{iter})$", title="Rank alignment")
    axes[1].plot(
        horizons,
        [float(score_lookup[f"short_rho_{horizon}_median"]["lofo_skill_vs_target_only"]) for horizon in horizons],
        marker="o",
        color=ORANGE,
    )
    axes[1].axhline(settings.reliable_lofo_skill, color=CHARCOAL, linestyle="--", linewidth=1.2, label="reliability gate")
    axes[1].set(xlabel="short rollout horizon m", ylabel="leave-family-out MAE skill", title="Cross-family calibration")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    name = "short_horizon_alignment.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)

    reuse = [row for row in eligible if row["family"] == "reuse"]
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    scatter = axis.scatter(
        [float(row["d_z"]) for row in reuse],
        [float(row["j_iter_median"]) for row in reuse],
        c=[abs(float(row["target_ratio"]) - float(row["source_ratio"])) for row in reuse],
        cmap="viridis",
        s=58,
        edgecolors="white",
        linewidths=0.6,
    )
    representative = _representative_disagreements(eligible)
    labeled_rows = representative[:2] + representative[-1:] if representative else []
    labeled_ids = {str(row["candidate_id"]) for row in labeled_rows}
    for row in reuse:
        if str(row["candidate_id"]) not in labeled_ids:
            continue
        axis.annotate(
            f"{float(row['source_ratio']):g}→{float(row['target_ratio']):g}",
            (float(row["d_z"]), float(row["j_iter_median"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axhline(0.0, color=LIGHT_GRAY, linestyle="--")
    axis.set(xlabel=r"right-space projector distance $d_Z$", ylabel=r"median $J_{iter}$")
    axis.set_title("Reused spaces: geometry versus solver value", loc="left")
    axis.grid(alpha=0.20)
    figure.colorbar(scatter, ax=axis, label=r"$|R_{donor}-R_{target}|$")
    name = "reuse_geometry_vs_solver_value.png"
    _save_figure(figure, plot_dir / name)
    names.append(name)
    return names


def _format(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _representative_disagreements(summaries: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    reuse = [row for row in summaries if row["family"] == "reuse" and row["failure"] == ""]
    if not reuse:
        return []
    d_threshold = float(np.quantile([float(row["d_z"]) for row in reuse], 0.60))
    useful = [row for row in reuse if float(row["d_z"]) >= d_threshold and float(row["j_iter_median"]) > 0.15]
    return sorted(useful, key=lambda row: (-float(row["j_iter_median"]), -float(row["d_z"])))[:6]


def write_report(
    output: Path,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    scorecard: Sequence[Mapping[str, Any]],
    block_summaries: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    settings: Stage14Settings,
) -> Path:
    """Write the requested concise, source-backed technical Markdown report."""

    family_counts = {
        family: sum(row["family"] == family for row in summaries)
        for family in ("oracle", "basis_rotation", *ANALYSIS_FAMILIES)
    }
    baseline_table = []
    for ratio in settings.target_ratios:
        local = [row for row in baseline_rows if float(row["target_ratio"]) == ratio]
        baseline_table.append(
            (
                _format(ratio, 0),
                _format(np.median([float(row["iterations"]) for row in local]), 0),
                _format(np.mean([bool(row["converged"]) for row in local]), 2),
                _format(max(float(row["true_final_relative_residual"]) for row in local)),
            )
        )
    score_table = [
        (
            row["metric_label"],
            _format(row["overall_spearman_vs_iterations"]),
            _format(row["overall_spearman_vs_j"]),
            _format(row["target_spearman_median"]),
            f"{int(row['consistent_family_count'])}/{int(row['available_family_count'])}",
            _format(row["lofo_skill_vs_target_only"]),
            _format(row["lofo_prediction_spearman"]),
            _format(row["reliable"]),
        )
        for row in scorecard
    ]
    corr_lookup = {(row["scope"], row["metric"]): row for row in correlations}
    family_table = [
        (
            metric_label,
            *(
                _format(corr_lookup[(f"family_{family}", metric)]["spearman_vs_j_iter"])
                for family in ANALYSIS_FAMILIES
            ),
        )
        for metric, metric_label, _ in METRICS
    ]
    block_table = [
        (
            _format(row["target_ratio"], 0),
            f"[{int(row['block_start'])},{int(row['block_stop'])})",
            _format(row["iteration_penalty_median"], 0),
            f"{int(row['iteration_penalty_min'])}…{int(row['iteration_penalty_max'])}",
            _format(row["normalized_weight"]),
            _format(row["rhs_sign_consistent"]),
        )
        for row in block_summaries
    ]
    disagreements = _representative_disagreements(summaries)
    disagreement_table = [
        (
            f"{float(row['source_ratio']):g}→{float(row['target_ratio']):g}",
            _format(row["d_z"]),
            _format(row["d_w"]),
            _format(row["short_rho_5_median"]),
            _format(row["iterations_median"], 0),
            _format(row["j_iter_median"]),
        )
        for row in disagreements
    ]
    rotations = [row for row in summaries if row["family"] == "basis_rotation"]
    rotation_dz = max([float(row["d_z"]) for row in rotations], default=float("nan"))
    rotation_dw = max([float(row["d_w"]) for row in rotations], default=float("nan"))
    oracle_iterations = {
        float(row["target_ratio"]): float(row["iterations_median"])
        for row in summaries
        if row["family"] == "oracle"
    }
    rotation_iteration_error = max(
        [abs(float(row["iterations_median"]) - oracle_iterations[float(row["target_ratio"])]) for row in rotations],
        default=float("nan"),
    )
    total_candidates = len(summaries)
    total_full_solves = len(candidate_rows)
    cap_fraction = float(np.mean([bool(row["cap_hit"]) for row in candidate_rows]))
    analysis_candidate_rows = [
        row for row in candidate_rows if row["family"] in ANALYSIS_FAMILIES
    ]
    short_ranges = {
        horizon: (
            min(float(row[f"short_rho_{horizon}"]) for row in analysis_candidate_rows),
            max(float(row[f"short_rho_{horizon}"]) for row in analysis_candidate_rows),
        )
        for horizon in settings.short_horizons
    }
    horizon = decision.get("recommended_short_horizon")
    if decision["conclusion"] == "C":
        network_target = (
            rf"Predict an $N\times q$ right-space embedding, orthonormalize it to $Z_\theta$, "
            rf"form $W_\theta=\operatorname{{orth}}(AZ_\theta)$, and optimize the {horizon}-step "
            "projected recycled-GMRES true-residual ratio. Do not regress directly to the unstable oracle basis."
        )
    elif decision["conclusion"] == "B":
        network_target = (
            "Predict a right space, but train through its target-side residual image using the "
            "basis-invariant $W$-projector loss; retain right-space geometry only as a diagnostic."
        )
    elif decision["conclusion"] == "A":
        network_target = "Predict the right space and use the basis-invariant $Z$-projector loss."
    else:
        network_target = "Do not train a network: none of the tested targets is a reliable proxy for final solver value."

    report = rf"""# Stage 14: solver-aware target and loss diagnostics

## Executive recommendation

**Decision {decision['conclusion']}: {decision['conclusion_label']}**

**Neural-training gate: {decision['neural_training_decision']}.** {network_target}

Among the tested proxies, residual/test-space geometry is the strongest *diagnostic* ($\rho_S(d_W,J_{{iter}})={float(next(row for row in scorecard if row['metric'] == 'd_w')['overall_spearman_vs_j']):.3f}$ overall; median within-$R$ {float(next(row for row in scorecard if row['metric'] == 'd_w')['target_spearman_median']):.3f}). It is not a reliable learning loss: it aligns in only {int(next(row for row in scorecard if row['metric'] == 'd_w')['consistent_family_count'])}/4 candidate families and its leave-family-out prediction correlation is only {float(next(row for row in scorecard if row['metric'] == 'd_w')['lofo_prediction_spearman']):.3f}. No short horizon repairs that failure.

This conclusion separates solver relevance from learnability. A metric can rank candidate spaces well without implying that the map from system parameters to that space is smooth. Stage 10–11 already found the individual and clustered Ritz labels geometrically unstable; this experiment asks only which observable is most faithful to held-out Krylov acceleration.

## 1. Exact fixed experiment

The [Stage 13 report](../stage13_persistent_preconditioner_vs_krylov_recycling/stage13_persistent_preconditioner_vs_krylov_recycling.md) and its implementation are the empirical and algorithmic starting points. The experiment keeps $N=5041$, node seed 7, jitter 0.15, stencil 25, 30° rotated anisotropy, $R\in\{{90,95,100,105,110\}}$, ILU drop tolerance $10^{{-3}}$, fill factor 5, COLAMD, zero diagonal-pivot threshold, right-preconditioned $B=AM^{{-1}}$, tolerance $10^{{-8}}$, restart 50, cap 300, construction RHS seed {settings.stage10.construction_rhs_seed}, held-out RHS seeds {settings.stage10.evaluation_rhs_seeds}, and $q=8$.

The oracle $Z_*(R)$ is the exact archived Stage 10/13 solution-space Ritz prefix. Every candidate is evaluated only with Stage 13 Method B: $AZ=WR$, $U=ZR^{{-1}}$, $AU=W$, initial projection against $W$, and Arnoldi on $(I-WW^T)AM^{{-1}}$. No solver, Ritz selection, ILU, or neural model is changed.

Strong-ILU controls:

{_table(['R', 'median iterations', 'convergence', 'max true residual'], baseline_table)}

R=100 and R=110 baseline iteration reductions are censored by the 300-step cap; $J_{{iter}}$ there is relative to the observed capped baseline, not an uncensored iteration count.

## 2. Candidate-space protocol

The library contains **{total_candidates}** candidate/target spaces and **{total_full_solves}** full held-out solves; each also receives an independent 10-step capped rollout. Candidate counts are: oracle {family_counts['oracle']}, invariant basis rotations {family_counts['basis_rotation']}, reuse {family_counts['reuse']}, interpolation {family_counts['interpolation']}, controlled random perturbation {family_counts['random_perturbation']}, and atomic-block replacement {family_counts['partial_block_replacement']}.

- **Reuse:** every other focal-ratio oracle is applied unchanged to each target, retaining both nearby and far donors.
- **Interpolation:** shortest Grassmann geodesics from the target oracle toward immediate neighbors, plus bracketing-neighbor geodesics for interior targets, at $t\in\{{0.25,0.5,0.75\}}$.
- **Basis rotations:** two random orthogonal changes of coordinates inside each oracle range. These must be identical under projector-based metrics and solver action.
- **Controlled perturbations:** all eight principal directions are moved by 5°, 15°, 30°, or 60° toward an independently sampled orthogonal complement, with two replicates.
- **Partial replacements:** one real Ritz direction or inseparable complex-pair block is replaced at a time. Complex pairs are never split.

The basis-invariance control passes: maximum $d_Z={rotation_dz:.2e}$, maximum $d_W={rotation_dw:.2e}$, and maximum median-iteration difference from the oracle is {rotation_iteration_error:.0f}.

## 3. Metrics and analysis grain

For orthonormal candidate and oracle bases,

$$d_Z=\|\widetilde Z\widetilde Z^T-Z_*Z_*^T\|_F,$$

$$\widetilde W=\operatorname{{orth}}(A\widetilde Z),\quad W_*=\operatorname{{orth}}(AZ_*),\quad d_W=\|\widetilde W\widetilde W^T-W_*W_*^T\|_F.$$

A separate capped rollout gives, without using the final iteration count,

$$\rho_m(\widetilde Z)=\frac{{\|r_m(\widetilde Z)\|_2}}{{\|r_0\|_2}},\qquad m\in\{{1,2,5,10\}}.$$

The final benefit is

$$J_{{iter}}=\frac{{k_{{ILU}}-k(\widetilde Z)}}{{k_{{ILU}}}}.$$

Correlations use one observation per candidate/target after taking the median across the three held-out RHSs, preventing RHS replication from inflating sample size. The table also reports the median within-target rank correlation, consistency across the four nontrivial candidate families, and a leave-one-family-out linear calibration test with target fixed effects. The latter compares metric+target prediction against target-only prediction.

## 4. Which metric predicts final Krylov value?

{_table(['metric', 'Spearman vs k', 'Spearman vs J', 'median within-R', 'families aligned', 'LOFO MAE skill', 'LOFO pred. rho', 'reliable'], score_table)}

The one- and two-step residual ratios occupy very narrow ranges ({short_ranges[1][0]:.3f}–{short_ranges[1][1]:.3f} and {short_ranges[2][0]:.3f}–{short_ranges[2][1]:.3f}); they mostly measure the common initial stagnation. By ten steps the range is informative ({short_ranges[10][0]:.3f}–{short_ranges[10][1]:.3f}), but its ordering still reverses on the random-perturbation family and fails leave-family-out calibration.

![Candidate metrics versus solver benefit](plots/metric_vs_solver_benefit.png)

Within-family Spearman correlations with $J_{{iter}}$:

{_table(['metric', 'reuse', 'interpolation', 'random perturb.', 'block replacement'], family_table)}

![Cross-family correlation heatmap](plots/family_correlation_heatmap.png)

![Short-horizon alignment](plots/short_horizon_alignment.png)

The predeclared reliability gate requires overall Spearman ≤−{settings.reliable_overall_spearman:.2f}, median within-target Spearman ≤−{settings.reliable_target_median_spearman:.2f}, alignment ≤−{settings.reliable_family_spearman:.2f} in at least {settings.reliable_family_count}/4 families, leave-family-out skill ≥{settings.reliable_lofo_skill:.2f}, and held-family prediction Spearman ≥{settings.reliable_lofo_spearman:.2f}. A short rollout wins only if it also exceeds the best geometric metric by {settings.solver_aware_margin:.2f} in rank correlation and {settings.lofo_margin:.2f} in LOFO skill.

## 5. Geometry–solver disagreements

The following donor spaces are geometrically distant within the reuse family yet retain positive solver value. They directly retest Stage 13's observation that geometric distance need not imply uselessness.

{_table(['donor→target', 'd_Z', 'd_W', 'rho_5', 'iterations', 'J_iter'], disagreement_table)}

![Reused-space geometry versus solver value](plots/reuse_geometry_vs_solver_value.png)

Across all candidate runs, {100*cap_fraction:.1f}% hit the 300-step cap. These points are retained as poor solver outcomes rather than removed; their exact uncensored iteration counts are unknown.

## 6. Optional solver-critical block weighting

Atomic real directions and complex-pair planes receive deletion weights $\omega_b=\max(0,\operatorname{{median}}[k(Z_*\setminus Z_b)-k(Z_*)])$. The weighted metric is

$$d_\omega^2=\frac{{\sum_b\omega_b\|(I-P_{{\widetilde Z}})Z_b\|_F^2}}{{\sum_b\omega_b\,\dim(Z_b)}}.$$

{_table(['R', 'atomic block', 'median penalty', 'RHS range', 'normalized weight', 'sign consistent'], block_table)}

There are {decision['block_rhs_sign_inconsistent_count']} of {decision['block_count']} blocks whose deletion-penalty sign changes across RHSs. The weighted metric is therefore treated as a diagnostic, not silently promoted to a supervised target.

## 7. Recommendation for Stage 15

- **Solver relevance:** {decision['conclusion_label']}
- **Cross-system predictability:** geometric oracle labels remain unstable by the independent Stage 10–11 evidence. This stage does not convert solver relevance into proof that $\xi\mapsto Z_*(\xi)$ is learnable.
- **Training-label cost:** $d_Z$ requires the expensive 64-step oracle Ritz construction; $d_W$ additionally requires $q$ target $A$-applications and QR; a short solver-aware loss requires $m$ applications of both $A$ and ILU per training example and eventually differentiation or a surrogate through the projected iteration.
- **What the network should learn:** {network_target}
- **Decision:** **{decision['neural_training_decision']}** — {decision['neural_training_scope']}.
- **Stage 15 recommendation:** if work continues, keep it non-neural and test a restart-scale or multi-RHS solver functional; do not train against $P_{{Z_*}}$, $P_{{W_*}}$, or any $m\leq10$ rollout from this evidence.

## 8. Limitations and reproducibility

- Conclusions are restricted to the five fixed-node, fixed-angle anisotropy systems, q=8, three held-out RHSs, and the declared candidate mechanisms.
- Geodesics that begin at the target oracle are calibration probes, not deployable predictors. Bracketing interpolation and donor reuse are the target-independent transfer probes.
- Spearman correlation tests ranking, not causation or learnability. Leave-family-out calibration is linear and intentionally simple.
- Capped iteration counts create ties and censor the worst outcomes; convergence flags and true residuals are preserved in raw data.
- This macOS NumPy runtime emitted floating-point-status warnings from BLAS-backed `matmul`, including for an independent finite random-matrix product. Every saved basis, coupling condition number, short residual, final residual, and solver metric was therefore checked independently for finiteness; candidate-basis orthogonality was also checked explicitly.
- Wall times are recorded but target selection is based on solver progress and iterations because timing noise is hardware-dependent.
- No neural network is trained, and the archived oracle Ritz construction is unchanged.

Raw evidence is in `baseline_ilu_results.csv`, `candidate_index.csv`, `candidate_spaces.npz`, `candidate_results.csv`, `candidate_summary.csv`, `block_deletion_results.csv`, `block_importance.csv`, `correlations.csv`, `lofo_predictions.csv`, `lofo_summary.csv`, `metric_scorecard.csv`, `decision.json`, and `validation.json`.
"""
    path = output / "stage14_solver_aware_target_loss_diagnostics.md"
    path.write_text(report, encoding="utf-8")
    return path


def validate_outputs(
    output: Path,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    settings: Stage14Settings,
) -> Dict[str, Any]:
    """Validate row grain, solver truth, invariance, and required evidence."""

    expected_baselines = len(settings.target_ratios) * len(settings.stage10.evaluation_rhs_seeds)
    expected_candidates = len(summaries) * len(settings.stage10.evaluation_rhs_seeds)
    unique_summary = {
        (float(row["target_ratio"]), str(row["candidate_id"])) for row in summaries
    }
    rotation_rows = [row for row in summaries if row["family"] == "basis_rotation"]
    oracle_lookup = {
        float(row["target_ratio"]): float(row["iterations_median"])
        for row in summaries
        if row["family"] == "oracle"
    }
    baseline_iteration_lookup = {
        (float(row["target_ratio"]), int(row["rhs_seed"])): int(row["iterations"])
        for row in baseline_rows
    }
    j_formula_error_max = max(
        (
            abs(
                float(row["j_iter"])
                - (
                    baseline_iteration_lookup[(float(row["target_ratio"]), int(row["rhs_seed"]))]
                    - int(row["iterations"])
                )
                / max(
                    baseline_iteration_lookup[(float(row["target_ratio"]), int(row["rhs_seed"]))],
                    1,
                )
            )
            for row in candidate_rows
            if row["setup_failure"] == ""
        ),
        default=float("nan"),
    )
    raw_groups: Dict[Tuple[float, str], List[Mapping[str, Any]]] = {}
    for row in candidate_rows:
        raw_groups.setdefault(
            (float(row["target_ratio"]), str(row["candidate_id"])), []
        ).append(row)
    summary_lookup = {
        (float(row["target_ratio"]), str(row["candidate_id"])): row
        for row in summaries
    }
    aggregation_iteration_error_max = max(
        (
            abs(
                float(summary_lookup[key]["iterations_median"])
                - float(np.median([int(row["iterations"]) for row in local]))
            )
            for key, local in raw_groups.items()
            if summary_lookup[key]["failure"] == ""
        ),
        default=float("nan"),
    )
    with np.load(output / "candidate_spaces.npz") as archive:
        archived_bases_finite = all(
            np.isfinite(np.asarray(archive[key], dtype=float)).all()
            for key in archive.files
        )
        archived_basis_orthogonality_max = max(
            (
                float(
                    np.linalg.norm(
                        np.asarray(archive[key]).T @ np.asarray(archive[key])
                        - np.eye(np.asarray(archive[key]).shape[1]),
                        ord="fro",
                    )
                )
                for key in archive.files
            ),
            default=float("nan"),
        )
    validation = {
        "assessment": "ready_to_share",
        "baseline_row_count": len(baseline_rows),
        "baseline_expected_row_count": expected_baselines,
        "candidate_row_count": len(candidate_rows),
        "candidate_expected_row_count": expected_candidates,
        "candidate_summary_count": len(summaries),
        "candidate_unique_key_count": len(unique_summary),
        "j_iter_formula_error_max": j_formula_error_max,
        "aggregation_iteration_median_error_max": aggregation_iteration_error_max,
        "all_j_iter_formulas_exact": bool(j_formula_error_max <= 1.0e-14),
        "all_iteration_aggregations_exact": bool(
            aggregation_iteration_error_max <= 1.0e-14
        ),
        "archived_bases_finite": bool(archived_bases_finite),
        "archived_basis_orthogonality_error_max": archived_basis_orthogonality_max,
        "all_archived_bases_orthonormal": bool(
            archived_bases_finite and archived_basis_orthogonality_max <= 1.0e-10
        ),
        "all_basis_rotations_projector_invariant": bool(
            all(float(row["d_z"]) <= 1.0e-10 and float(row["d_w"]) <= 1.0e-10 for row in rotation_rows)
        ),
        "all_basis_rotations_solver_invariant": bool(
            all(
                abs(float(row["iterations_median"]) - oracle_lookup[float(row["target_ratio"])]) <= 1.0
                for row in rotation_rows
            )
        ),
        "all_short_rollouts_separate_from_full_solve": bool(
            all(
                bool(row["short_rollout_independent"])
                and int(row["short_rollout_max_iterations"]) == max(settings.short_horizons)
                and int(row["full_solve_max_iterations"]) == settings.stage10.solver.max_iterations
                for row in candidate_rows
                if row["setup_failure"] == ""
            )
        ),
        "all_candidate_metrics_finite": bool(
            all(
                all(
                    np.isfinite(float(row[field]))
                    for field in (
                        "d_z",
                        "d_w",
                        "condition_wtaz",
                        "short_rho_1",
                        "short_rho_2",
                        "short_rho_5",
                        "short_rho_10",
                        "true_final_relative_residual",
                        "j_iter",
                    )
                )
                for row in candidate_rows
                if row["setup_failure"] == ""
            )
        ),
        "all_short_arnoldi_checks_pass": bool(
            all(
                float(row["short_projected_arnoldi_relation_error"]) <= 1.0e-10
                for row in candidate_rows
                if row["setup_failure"] == ""
            )
        ),
        "all_converged_true_residuals_meet_tolerance": bool(
            all(
                (not bool(row["converged"]))
                or float(row["true_final_relative_residual"]) <= 1.05 * settings.stage10.solver.relative_tolerance
                for row in candidate_rows
            )
        ),
        "all_required_artifacts_present": all(
            (output / name).exists()
            for name in (
                "baseline_ilu_results.csv",
                "candidate_index.csv",
                "candidate_spaces.npz",
                "candidate_results.csv",
                "candidate_summary.csv",
                "block_deletion_results.csv",
                "block_importance.csv",
                "correlations.csv",
                "lofo_predictions.csv",
                "lofo_summary.csv",
                "metric_scorecard.csv",
                "decision.json",
            )
        ),
    }
    blockers = []
    if len(baseline_rows) != expected_baselines:
        blockers.append("baseline_row_count")
    if len(candidate_rows) != expected_candidates:
        blockers.append("candidate_row_count")
    if len(summaries) != len(unique_summary):
        blockers.append("candidate_key_uniqueness")
    for key in (
        "all_basis_rotations_projector_invariant",
        "all_basis_rotations_solver_invariant",
        "all_archived_bases_orthonormal",
        "all_short_rollouts_separate_from_full_solve",
        "all_candidate_metrics_finite",
        "all_short_arnoldi_checks_pass",
        "all_j_iter_formulas_exact",
        "all_iteration_aggregations_exact",
        "all_converged_true_residuals_meet_tolerance",
        "all_required_artifacts_present",
    ):
        if not validation[key]:
            blockers.append(key)
    validation["blockers"] = blockers
    validation["assessment"] = "ready_to_share" if not blockers else "needs_revision"
    _write_json(output / "validation.json", validation)
    if blockers:
        raise RuntimeError(f"Stage 14 validation failed: {blockers}")
    return validation


def run_stage14(output: Path) -> Path:
    """Run the complete fixed-protocol Stage 14 target/loss diagnostic."""

    settings = Stage14Settings()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "settings.json", asdict(settings))
    _write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "operator": "right-preconditioned A M^-1",
            "solver": "Stage 13 fixed-space recycled/projected GMRES",
            "neural_network_trained": False,
        },
    )
    print("[1/9] rebuilding fixed Stage 13 systems and exact archived Ritz spaces", flush=True)
    contexts, spaces, factors, _ = build_fixed_contexts(output, settings.stage13)
    eigenvalues = _load_eigenvalues(output, settings.target_ratios)
    print("[2/9] running unchanged strong-ILU baselines", flush=True)
    baseline_rows, baseline_lookup = run_baselines(output, contexts, factors, settings)
    print("[3/9] generating and evaluating candidate-space library", flush=True)
    candidates, blocks, candidate_rows, summaries = run_candidate_library(
        output, contexts, spaces, eigenvalues, baseline_lookup, settings
    )
    print("[4/9] computing block-level deletion importance", flush=True)
    _, block_summaries, weights = run_block_importance(
        output, contexts, spaces, blocks, candidate_rows, settings
    )
    print("[5/9] attaching weighted geometry", flush=True)
    attach_weighted_geometry(candidates, blocks, weights, spaces, candidate_rows, summaries)
    _write_csv(output / "candidate_results.csv", candidate_rows)
    _write_csv(output / "candidate_summary.csv", summaries)
    print("[6/9] computing rank correlations and leave-family-out tests", flush=True)
    correlations = compute_correlations(output, summaries, candidate_rows, settings)
    _, lofo = leave_one_family_out(output, summaries)
    scorecard = build_metric_scorecard(output, correlations, lofo, settings)
    decision = make_decision(output, scorecard, block_summaries, settings)
    print("[7/9] generating research figures", flush=True)
    make_plots(output, summaries, correlations, scorecard, settings)
    print("[8/9] writing report", flush=True)
    report = write_report(
        output,
        baseline_rows,
        candidate_rows,
        summaries,
        correlations,
        scorecard,
        block_summaries,
        decision,
        settings,
    )
    print("[9/9] validating evidence", flush=True)
    validate_outputs(output, baseline_rows, candidate_rows, summaries, settings)
    return report
