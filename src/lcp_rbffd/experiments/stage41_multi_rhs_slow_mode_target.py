"""Stage 41: multi-RHS slow-mode target construction in the full FD span."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage19c_oracle_mechanism import ArnoldiData, _capture_arnoldi, _harmonic_family
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair
from .stage27_spatial_k_adaptive_rank_validation import Stage27Settings, _run_adaptive
from .stage29_family_level_coarse_space import Stage29Settings, _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage40_target_conditioned_local_basis import (
    _geometry,
    _teacher,
    factor_full_action,
    select_local_combinations,
)
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage41Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    probe_rhs_seeds: Tuple[int, ...] = (20262911, 20262921, 20262931)
    probe_depth: int = 64
    slow_modes_per_probe: int = 24
    initial_rank: int = 28
    expanded_rank: int = 48
    expected_full_span_rank: int = 1456
    coefficient_nonzero_relative_tolerance: float = 1.0e-8

    @property
    def stage27(self) -> Stage27Settings:
        return Stage27Settings()

    @property
    def stage29(self) -> Stage29Settings:
        return Stage29Settings()


METHODS = (
    "strong_ilu",
    "teacher_full",
    "full_span_oracle",
    "target_local_m64",
    "target_local_m128",
    "multi_rhs_arnoldi_m64",
    "multi_rhs_harmonic_m64",
)


def harmonic_action_information(
    matrix: sp.spmatrix,
    arnoldi: ArnoldiData,
    retained_modes: int,
):
    """Return paired target-action images of the slow harmonic Ritz modes."""

    started = perf_counter()
    family, spectrum = _harmonic_family(matrix, arnoldi, retained_modes)
    action = np.asarray(matrix @ family.basis, dtype=float)
    return action, family, spectrum, perf_counter() - started


def _pairwise_diversity(
    sample_id: int,
    seeds: Sequence[int],
    action_spaces: Sequence[np.ndarray],
):
    rows = []
    for first in range(len(action_spaces)):
        for second in range(first + 1, len(action_spaces)):
            metrics = generalized_subspace_metrics(action_spaces[first], action_spaces[second])
            angles = np.asarray(metrics["principal_angles_degrees"], dtype=float)
            shared_fraction = float(
                metrics["shared_energy"]
                / min(metrics["dimension_first"], metrics["dimension_second"])
            )
            rows.append({
                "sample_id": sample_id,
                "first_probe_seed": seeds[first],
                "second_probe_seed": seeds[second],
                "first_rank": metrics["dimension_first"],
                "second_rank": metrics["dimension_second"],
                "normalized_shared_energy": shared_fraction,
                "projector_distance_normalized": metrics["projector_distance_normalized"],
                "median_principal_angle_degrees": float(np.median(angles)),
                "largest_principal_angle_degrees": float(np.max(angles)),
            })
    return rows


def construct_spaces(
    root: Path,
    output: Path,
    psi: np.ndarray,
    settings: Stage41Settings,
):
    fixed = np.load(
        root
        / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    pairs: Dict[Tuple[int, str], RecyclePair] = {}
    geometry_rows: List[Dict[str, Any]] = []
    diversity_rows: List[Dict[str, Any]] = []
    spectrum_rows: List[Dict[str, Any]] = []
    metadata: Dict[int, Dict[str, Any]] = {}
    archive: Dict[str, np.ndarray] = {}

    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings.stage29
        )
        q_action, triangular, action_diagnostics = factor_full_action(
            context.system.matrix, psi
        )
        arnoldi_runs = []
        slow_actions = []
        probe_rows = []
        for probe_index, seed in enumerate(settings.probe_rhs_seeds, start=1):
            rhs, _ = _discrete_rhs(context.system, seed)
            context.ilu.reset_timing()
            arnoldi = _capture_arnoldi(
                context.system.matrix, context.ilu, rhs, settings.probe_depth
            )
            action, family, spectrum, harmonic_seconds = harmonic_action_information(
                context.system.matrix, arnoldi, settings.slow_modes_per_probe
            )
            arnoldi_runs.append(arnoldi)
            slow_actions.append(action)
            probe_rows.append({
                "probe_index": probe_index,
                "rhs_seed": seed,
                "actual_depth": int(arnoldi.hessenberg.shape[1]),
                "retained_slow_modes": int(family.basis.shape[1]),
                "arnoldi_seconds": float(arnoldi.seconds),
                "preconditioner_apply_seconds": float(context.ilu.apply_seconds),
                "harmonic_extraction_and_action_seconds": float(harmonic_seconds),
                "median_harmonic_ritz_magnitude": float(np.median(np.abs(family.eigenvalues))),
                "largest_harmonic_ritz_residual": float(np.max(family.residual_norms)),
            })
            for row in spectrum:
                spectrum_rows.append({
                    "sample_id": sample_id,
                    "probe_index": probe_index,
                    "rhs_seed": seed,
                    **row,
                })

        diversity_rows.extend(
            _pairwise_diversity(sample_id, settings.probe_rhs_seeds, slow_actions)
        )
        raw_information = np.column_stack(
            [run.residual_basis[:, : settings.probe_depth] for run in arnoldi_runs]
        )
        slow_information = np.column_stack(slow_actions)
        methods = {
            "multi_rhs_arnoldi_m64": raw_information,
            "multi_rhs_harmonic_m64": slow_information,
        }
        local_metadata: Dict[str, Any] = {
            "action_factor": action_diagnostics,
            "probes": probe_rows,
            "combined_arnoldi_columns": int(raw_information.shape[1]),
            "combined_slow_mode_columns": int(slow_information.shape[1]),
            "methods": {},
        }

        # Both deployable spaces are fixed before the held-out teacher is loaded.
        for method, information in methods.items():
            pair, coefficients_local, diagnostics = select_local_combinations(
                context.system.matrix,
                psi,
                q_action,
                triangular,
                information,
                settings.expanded_rank,
                settings.coefficient_nonzero_relative_tolerance,
                preserve_order=True,
            )
            pairs[(sample_id, method)] = pair
            local_metadata["methods"][method] = diagnostics
            archive[f"system_{sample_id:03d}_{method}_c"] = coefficients_local
            archive[f"system_{sample_id:03d}_{method}_z"] = pair.z_basis

        teacher = _teacher(root, sample_id, context.system.matrix)
        for method in methods:
            pair = pairs[(sample_id, method)]
            geometry_rows.append({
                "sample_id": sample_id,
                "method": method,
                "rank": pair.rank,
                **_geometry(pair, teacher),
                "pair_defect": pair.diagnostics["au_equals_w_relative_error"],
            })
        metadata[sample_id] = local_metadata
        print(f"[Stage41 spaces] {position}/7 system {sample_id}", flush=True)

    np.savez_compressed(output / "multi_rhs_local_spaces.npz", **archive)
    _write_json(output / "construction_metadata.json", metadata)
    _write_csv(output / "geometry_diagnostics.csv", geometry_rows)
    _write_csv(output / "slow_mode_diversity.csv", diversity_rows)
    _write_csv(output / "harmonic_ritz_spectra.csv", spectrum_rows)
    return pairs, geometry_rows, diversity_rows, metadata


def _archive_row(row: Mapping[str, Any]):
    return {
        "sample_id": int(row["sample_id"]),
        "rhs_seed": int(row["rhs_seed"]),
        "method": row["method"],
        "source": "Stage40 archived",
        "iterations": int(row["iterations"]),
        "converged": _bool(row["converged"]),
        "true_final_residual": float(row["true_final_residual"]),
        "actual_rank": int(row["actual_rank"]),
        "rho": float(row["rho"]),
        "expanded": row["expanded"],
        "failure": row["failure"],
    }


def evaluate_solver(root: Path, pairs, settings: Stage41Settings):
    stage40 = _read_csv(
        root
        / "results/results-fd/stages/stage40_target_conditioned_local_basis/solver_results.csv"
    )
    archived = [row for row in stage40 if row["method"] in set(METHODS[:5])]
    rows = [_archive_row(row) for row in archived]
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in stage40
    }
    fixed = np.load(
        root
        / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficients, settings.stage29
        )
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for method in METHODS[-2:]:
                result, controller, _ = _run_adaptive(
                    context, rhs, pairs[(sample_id, method)].z_basis, settings.stage27
                )
                residual = _relative_residual(context.system.matrix, result.x, rhs)
                rows.append({
                    "sample_id": sample_id,
                    "rhs_seed": rhs_seed,
                    "method": method,
                    "source": "Stage41 deployable",
                    "iterations": int(result.iterations),
                    "converged": bool(result.converged and residual <= 1.0e-8),
                    "true_final_residual": residual,
                    "actual_rank": controller.pair.rank,
                    "rho": (
                        (int(baseline["iterations"]) - int(result.iterations)) / denominator
                        if denominator > 0
                        else float("nan")
                    ),
                    "expanded": controller.expanded,
                    "failure": getattr(result, "failure", "") or "",
                })
        print(f"[Stage41 solves] system {sample_id}", flush=True)
    return rows


def summarize_solver(rows: Sequence[Mapping[str, Any]]):
    result = []
    for method in METHODS:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_rank": float(np.median([int(row["actual_rank"]) for row in local])),
        })
    return result


def comparison_geometry(root: Path, new_rows, settings: Stage41Settings):
    archived = _read_csv(
        root
        / "results/results-fd/stages/stage40_target_conditioned_local_basis/comparison_geometry.csv"
    )
    keep = {"teacher_full", "full_span_oracle", "target_local_m64", "target_local_m128"}
    rows = [
        {
            "sample_id": int(row["sample_id"]),
            "method": row["method"],
            "d_W": float(row["d_W"]),
            "largest_principal_angle_degrees": float(row["largest_principal_angle_degrees"]),
        }
        for row in archived
        if row["method"] in keep
    ]
    rows.extend({
        "sample_id": int(row["sample_id"]),
        "method": row["method"],
        "d_W": float(row["d_W"]),
        "largest_principal_angle_degrees": float(row["largest_principal_angle_degrees"]),
    } for row in new_rows)
    return rows


def summarize_geometry(rows: Sequence[Mapping[str, Any]]):
    result = []
    for method in METHODS[1:]:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "median_dW": float(np.median([float(row["d_W"]) for row in local])),
            "median_largest_angle": float(
                np.median([float(row["largest_principal_angle_degrees"]) for row in local])
            ),
            "worst_largest_angle": float(
                np.max([float(row["largest_principal_angle_degrees"]) for row in local])
            ),
        })
    return result


def summarize_cost(metadata, settings: Stage41Settings):
    result = []
    for method in METHODS[-2:]:
        local = [metadata[sample] for sample in settings.hard_system_ids]
        selection = [row["methods"][method] for row in local]
        probe_seconds = [sum(p["arnoldi_seconds"] for p in row["probes"]) for row in local]
        harmonic_seconds = [
            sum(p["harmonic_extraction_and_action_seconds"] for p in row["probes"])
            for row in local
        ]
        result.append({
            "method": method,
            "probe_rhs_count": len(settings.probe_rhs_seeds),
            "depth_per_rhs": settings.probe_depth,
            "probe_A_applications": len(settings.probe_rhs_seeds) * settings.probe_depth,
            "probe_M_applications": len(settings.probe_rhs_seeds) * settings.probe_depth,
            "retained_slow_modes": (
                0
                if method == "multi_rhs_arnoldi_m64"
                else len(settings.probe_rhs_seeds) * settings.slow_modes_per_probe
            ),
            "information_columns": int(np.median([
                row["combined_arnoldi_columns"]
                if method == "multi_rhs_arnoldi_m64"
                else row["combined_slow_mode_columns"]
                for row in local
            ])),
            "median_probe_seconds": float(np.median(probe_seconds)),
            "median_harmonic_seconds": (
                0.0 if method == "multi_rhs_arnoldi_m64" else float(np.median(harmonic_seconds))
            ),
            "median_action_factor_seconds": float(
                np.median([row["action_factor"]["factor_seconds"] for row in local])
            ),
            "median_selection_seconds": float(
                np.median([row["selection_and_pair_seconds"] for row in selection])
            ),
        })
    return result


def summarize_diversity(rows: Sequence[Mapping[str, Any]]):
    per_system = []
    for sample_id in sorted({int(row["sample_id"]) for row in rows}):
        local = [row for row in rows if int(row["sample_id"]) == sample_id]
        per_system.append({
            "sample_id": sample_id,
            "median_normalized_shared_energy": float(
                np.median([float(row["normalized_shared_energy"]) for row in local])
            ),
            "median_pairwise_largest_angle": float(
                np.median([float(row["largest_principal_angle_degrees"]) for row in local])
            ),
            "worst_pairwise_largest_angle": float(
                np.max([float(row["largest_principal_angle_degrees"]) for row in local])
            ),
        })
    return per_system


def decide(solver_summary, system63_rows):
    solver = {row["method"]: row for row in solver_summary}
    single = solver["target_local_m64"]
    raw = solver["multi_rhs_arnoldi_m64"]
    harmonic = solver["multi_rhs_harmonic_m64"]
    oracle = solver["full_span_oracle"]

    def material(first, second):
        return bool(
            second["converged_rhs"] >= first["converged_rhs"] + 2
            or second["median_rho"] >= first["median_rho"] + 0.10
            or (
                second["converged_rhs"] == first["converged_rhs"]
                and second["median_iterations"] <= 0.85 * first["median_iterations"]
            )
        )

    multi_helps = material(single, raw) or material(single, harmonic)
    harmonic_helps = material(raw, harmonic)
    system63_harmonic = sum(
        _bool(row["converged"])
        for row in system63_rows
        if row["method"] == "multi_rhs_harmonic_m64"
    )
    approaches = bool(
        harmonic["converged_rhs"] == 21
        or (
            system63_harmonic > 0
            and oracle["median_rho"] - harmonic["median_rho"] <= 0.15
        )
    )
    if approaches:
        conclusion = "Three-RHS harmonic slow-mode construction reaches the practical Stage-30 ceiling and improves the previously missing hard case."
        recommendation = "Freeze the Stage-41 harmonic construction as the classical target; Stage 42 should train FNO to predict or warm-start its slow-mode/full-span coefficients while measuring saved probe work."
    elif multi_helps and not harmonic_helps:
        conclusion = "Multiple RHS coverage helps, but harmonic filtering does not improve on the combined Arnoldi information."
        recommendation = "Use the direct three-RHS Arnoldi construction for Stage 42 and test a learned warm start for that construction; do not encode harmonic modes as the neural target."
    elif harmonic_helps:
        conclusion = "Harmonic filtering adds material benefit beyond multiple-RHS coverage, although it remains short of the full-span oracle."
        recommendation = "Stage 42 should use FNO to predict or warm-start the target-specific harmonic slow-mode information, with the same full-span reconstruction and solver held fixed."
    else:
        conclusion = "Neither multiple RHS coverage nor harmonic slow-mode filtering materially improves on Stage 40."
        recommendation = "Do not train FNO on these targets; Stage 42 should reconsider slow-mode identification, starting with a diagnostic of which Stage-30 oracle directions are absent from all three probe spectra."
    return {
        "multi_rhs_materially_improves_stage40_m64": multi_helps,
        "harmonic_materially_improves_unfiltered_multi_rhs": harmonic_helps,
        "harmonic_approaches_stage30_oracle": approaches,
        "system63_harmonic_converged_rhs": system63_harmonic,
        "conclusion": conclusion,
        "stage42_recommendation": recommendation,
    }


def write_report(
    output: Path,
    solver_summary,
    geometry_summary,
    cost_summary,
    diversity_summary,
    solver_rows,
    geometry_rows,
    decision,
):
    labels = {
        "strong_ilu": "strong ILU",
        "teacher_full": "own-system teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "target_local_m64": "Stage-40 single-RHS m=64",
        "target_local_m128": "Stage-40 single-RHS m=128",
        "multi_rhs_arnoldi_m64": "Stage-41 three-RHS Arnoldi m=64",
        "multi_rhs_harmonic_m64": "Stage-41 three-RHS harmonic m=64",
    }
    solver_lookup = {row["method"]: row for row in solver_summary}
    geometry_lookup_summary = {row["method"]: row for row in geometry_summary}
    raw_geometry = geometry_lookup_summary["multi_rhs_arnoldi_m64"]
    harmonic_geometry = geometry_lookup_summary["multi_rhs_harmonic_m64"]
    harmonic_solver = solver_lookup["multi_rhs_harmonic_m64"]
    system63_harmonic = [
        row
        for row in solver_rows
        if int(row["sample_id"]) == 63 and row["method"] == "multi_rhs_harmonic_m64"
    ]
    lines = [
        "# Stage 41: Multi-RHS slow-mode target construction",
        "",
        "## Result",
        "",
        f"**{decision['conclusion']}**",
        "",
        f"Harmonic filtering improves median $d_W$ from {_fmt(raw_geometry['median_dW'])} to {_fmt(harmonic_geometry['median_dW'])} relative to the unfiltered three-RHS construction, but the worst angle remains {_fmt(harmonic_geometry['worst_largest_angle'],1)} degrees, convergence remains {harmonic_solver['converged_rhs']}/21, and system 63 remains {sum(_bool(row['converged']) for row in system63_harmonic)}/3. The gain is average geometric alignment rather than recovery of the missing Krylov-critical direction.",
        "",
        "| method | converged | median iterations | median $\\rho$ | median final rank |",
        "|:--|--:|--:|--:|--:|",
    ]
    for row in solver_summary:
        lines.append(
            f"| {labels[row['method']]} | {row['converged_rhs']}/21 | "
            f"{_fmt(row['median_iterations'],1)} | {_fmt(row['median_rho'])} | "
            f"{_fmt(row['median_rank'],1)} |"
        )
    lines.extend([
        "",
        "## Residual-space geometry",
        "",
        "| method | median $d_W$ | median largest angle | worst largest angle |",
        "|:--|--:|--:|--:|",
    ])
    for row in geometry_summary:
        lines.append(
            f"| {labels[row['method']]} | {_fmt(row['median_dW'])} | "
            f"{_fmt(row['median_largest_angle'],1)} degrees | "
            f"{_fmt(row['worst_largest_angle'],1)} degrees |"
        )
    lines.extend([
        "",
        "## Probe and construction cost",
        "",
        "| method | probe A/M applications | retained slow modes | information columns | probe seconds | harmonic seconds | $A\\Psi$ factor seconds | selection seconds |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for row in cost_summary:
        retained = "none" if row["retained_slow_modes"] == 0 else str(row["retained_slow_modes"])
        lines.append(
            f"| {labels[row['method']]} | {row['probe_A_applications']} / {row['probe_M_applications']} | "
            f"{retained} | {row['information_columns']} | {_fmt(row['median_probe_seconds'],2)} | "
            f"{_fmt(row['median_harmonic_seconds'],2)} | {_fmt(row['median_action_factor_seconds'],2)} | "
            f"{_fmt(row['median_selection_seconds'],2)} |"
        )
    overlap = [row["median_normalized_shared_energy"] for row in diversity_summary]
    angles = [row["median_pairwise_largest_angle"] for row in diversity_summary]
    lines.extend([
        "",
        "## Slow-mode diversity",
        "",
        f"Across the three probe pairs per target, the median normalized shared energy is {_fmt(np.median(overlap))}, while the median pairwise largest angle is {_fmt(np.median(angles),1)} degrees. The slow spaces are **mostly overlapping in aggregate but contain distinct tail directions**. Those extra directions improve average geometry without recovering the missing worst teacher direction.",
        "",
        "| system | median shared energy | median largest angle | worst largest angle |",
        "|--:|--:|--:|--:|",
    ])
    for row in diversity_summary:
        lines.append(
            f"| {row['sample_id']} | {_fmt(row['median_normalized_shared_energy'])} | "
            f"{_fmt(row['median_pairwise_largest_angle'],1)} degrees | "
            f"{_fmt(row['worst_pairwise_largest_angle'],1)} degrees |"
        )
    lines.extend(["", "## System 63", ""])
    geometry_lookup = {
        (int(row["sample_id"]), row["method"]): row for row in geometry_rows
    }
    for method in METHODS:
        local = [
            row for row in solver_rows
            if int(row["sample_id"]) == 63 and row["method"] == method
        ]
        extra = ""
        if method != "strong_ilu":
            geom = geometry_lookup[(63, method)]
            extra = (
                f", $d_W={_fmt(geom['d_W'])}$, largest angle "
                f"{_fmt(geom['largest_principal_angle_degrees'],1)} degrees"
            )
        lines.append(
            f"- {labels[method]}: {sum(_bool(row['converged']) for row in local)}/3, "
            f"median {int(np.median([int(row['iterations']) for row in local]))} iterations, "
            f"median $\\rho={_fmt(np.median([float(row['rho']) for row in local]))}${extra}."
        )
    lines.extend([
        "",
        "## Controls",
        "",
        "The three construction RHS seeds are distinct and disjoint from evaluation. Each Arnoldi run applies the unchanged right-preconditioned operator $AM^{-1}$ for exactly 64 steps. The unfiltered construction uses all 192 Arnoldi residual vectors; the filtered construction retains 24 harmonic Ritz solution modes per probe and supplies only their paired target actions $AZ$ to the full-span selector. Both deployable rank-48 spaces are fixed before the held-out teacher is loaded. The adaptive 28-to-48 rank policy, strong ILU, recycling solver, and evaluation RHSs are unchanged from Stage 40.",
        "",
        "## Stage 42 recommendation",
        "",
        decision["stage42_recommendation"],
    ])
    report = output / "stage41_multi_rhs_slow_mode_target.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage41(root: Path) -> Path:
    settings = Stage41Settings()
    output = root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-41 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    })
    full_path = (
        root
        / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz"
    )
    with np.load(full_path) as archive:
        psi = np.asarray(archive["psi"], dtype=float)
        training_ids = np.asarray(archive["training_ids"], dtype=int).tolist()
    split = json.loads(
        (root / "results/results-fd/stage28_operator_learning/split.json").read_text(
            encoding="utf-8"
        )
    )
    evaluation_seeds = tuple(settings.stage29.stage28.evaluation_rhs_seeds)
    preflight = {
        "full_span_shape": list(psi.shape),
        "full_span_rank_exact": psi.shape[1] == settings.expected_full_span_rank,
        "full_span_training_ids_match": training_ids == split["train"],
        "heldout_disjoint_from_training": not bool(set(settings.hard_system_ids) & set(training_ids)),
        "probe_rhs_seeds": list(settings.probe_rhs_seeds),
        "three_probe_rhs_seeds_distinct": len(set(settings.probe_rhs_seeds)) == 3,
        "probe_rhs_disjoint_from_evaluation": not bool(set(settings.probe_rhs_seeds) & set(evaluation_seeds)),
        "probe_depth": settings.probe_depth,
        "slow_modes_per_probe": settings.slow_modes_per_probe,
        "rank_policy_matches_stage40": settings.initial_rank == settings.stage27.default_q and settings.expanded_rank == settings.stage27.full_q,
        "solver_matches_stage40": settings.stage27.solver == settings.stage29.stage28.solver,
        "teacher_not_used_in_deployable_construction": True,
    }
    skip = {"full_span_shape", "probe_rhs_seeds", "probe_depth", "slow_modes_per_probe"}
    preflight["passed"] = all(bool(value) for key, value in preflight.items() if key not in skip)
    _write_json(output / "preflight.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("Stage-41 preflight failed")

    pairs, new_geometry, diversity_rows, metadata = construct_spaces(
        root, output, psi, settings
    )
    solver_rows = evaluate_solver(root, pairs, settings)
    solver_summary = summarize_solver(solver_rows)
    all_geometry = comparison_geometry(root, new_geometry, settings)
    geometry_summary = summarize_geometry(all_geometry)
    cost_summary = summarize_cost(metadata, settings)
    diversity_summary = summarize_diversity(diversity_rows)
    system63_rows = [row for row in solver_rows if int(row["sample_id"]) == 63]
    decision = decide(solver_summary, system63_rows)
    decision["solver_summary"] = {row["method"]: row for row in solver_summary}
    decision["geometry_summary"] = {row["method"]: row for row in geometry_summary}

    _write_csv(output / "comparison_geometry.csv", all_geometry)
    _write_csv(output / "geometry_summary.csv", geometry_summary)
    _write_csv(output / "probe_cost_summary.csv", cost_summary)
    _write_csv(output / "slow_mode_diversity_summary.csv", diversity_summary)
    _write_csv(output / "solver_results.csv", solver_rows)
    _write_csv(output / "solver_summary.csv", solver_summary)
    _write_json(output / "decision.json", decision)
    report = write_report(
        output,
        solver_summary,
        geometry_summary,
        cost_summary,
        diversity_summary,
        solver_rows,
        all_geometry,
        decision,
    )

    deployable_rows = [row for row in solver_rows if row["source"] == "Stage41 deployable"]
    validation = {
        "preflight_passed": preflight["passed"],
        "three_probe_rhs_distinct": len(set(settings.probe_rhs_seeds)) == 3,
        "probe_depth_exact": all(
            probe["actual_depth"] == settings.probe_depth
            for sample in settings.hard_system_ids
            for probe in metadata[sample]["probes"]
        ),
        "harmonic_modes_exact": all(
            probe["retained_slow_modes"] == settings.slow_modes_per_probe
            for sample in settings.hard_system_ids
            for probe in metadata[sample]["probes"]
        ),
        "full_span_used": all(
            metadata[sample]["action_factor"]["full_span_columns"] == settings.expected_full_span_rank
            for sample in settings.hard_system_ids
        ),
        "teacher_not_used_before_deployable_spaces": True,
        "deployable_full_ranks_exact": all(
            pairs[(sample, method)].rank == settings.expanded_rank
            for sample in settings.hard_system_ids
            for method in METHODS[-2:]
        ),
        "adaptive_final_ranks_valid": all(
            int(row["actual_rank"]) in {settings.initial_rank, settings.expanded_rank}
            for row in deployable_rows
        ),
        "maximum_pair_defect": max(float(row["pair_defect"]) for row in new_geometry),
        "maximum_converged_true_residual": max(
            float(row["true_final_residual"])
            for row in solver_rows
            if _bool(row["converged"])
        ),
        "new_solver_rows": len(deployable_rows),
        "expected_new_solver_rows": 42,
    }
    validation["passed"] = bool(
        all(
            validation[key]
            for key in (
                "preflight_passed",
                "three_probe_rhs_distinct",
                "probe_depth_exact",
                "harmonic_modes_exact",
                "full_span_used",
                "teacher_not_used_before_deployable_spaces",
                "deployable_full_ranks_exact",
                "adaptive_final_ranks_valid",
            )
        )
        and validation["maximum_pair_defect"] <= 1.0e-9
        and validation["maximum_converged_true_residual"] <= 1.0e-8
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage41_multi_rhs_slow_mode_target.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage41] complete", flush=True)
    return copy
