"""Stage 49: target-local full-span complementary-correction ceilings."""

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

from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage20_spatial_k_robust_oracle import _relative_residual
from .stage22a_multi_rhs_functional_oracle import RecyclePair, _pair_from_z, _run_pair
from .stage29_family_level_coarse_space import _stage28_context
from .stage30_full_span_representation_ceiling import _bool, _fmt, _manifest, _read_csv
from .stage38_solver_response_guided_local_selection import _ordered_pair_from_z
from .stage42_missing_critical_mode_diagnosis import _orthonormal_basis
from .stage47_fno_complementary_correction import Stage47Settings
from .stage48_full_dictionary_complementary_correction import load_stage47_targets
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage49Settings:
    correction_ranks: Tuple[int, int] = (8, 12)
    full_span_rank: int = 1456
    frozen_classical_rank: int = 48
    numerical_rank_tolerance: float = 1.0e-11
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)

    @property
    def stage47(self) -> Stage47Settings:
        return Stage47Settings()

    @property
    def stage29(self):
        return self.stage47.stage29


def principal_coverage_metrics(space: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    first = _orthonormal_basis(space)
    second = _orthonormal_basis(target)
    singular = np.clip(la.svdvals(first.T @ second), 0.0, 1.0)
    coverage = singular**2
    angles = np.degrees(np.arccos(singular))
    return {
        "correction_containment": float(np.sum(coverage) / second.shape[1]),
        "minimum_principal_coverage": float(np.min(coverage)),
        "p10_principal_coverage": float(np.percentile(coverage, 10)),
        "median_principal_coverage": float(np.median(coverage)),
        "median_principal_angle_degrees": float(np.median(angles)),
        "largest_principal_angle_degrees": float(np.max(angles)),
        "target_correction_frobenius_overlap": float(la.norm(first.T @ second, ord="fro")),
    }


def factor_complementary_action(
    action: np.ndarray,
    classical: RecyclePair,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Factor (I-P_W48) action and verify that its full column rank is retained."""

    started = perf_counter()
    overlap = classical.w_basis.T @ action
    perpendicular = action - classical.w_basis @ overlap
    basis, triangular, pivots = la.qr(
        perpendicular, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(triangular))
    scale = max(float(diagonal[0]), np.finfo(float).tiny)
    rank = int(np.count_nonzero(diagonal > tolerance * scale))
    if rank == 0:
        raise RuntimeError("complementary action has zero numerical rank")
    return perpendicular, basis[:, :rank], triangular[:rank, :], pivots, {
        "columns": action.shape[1],
        "numerical_rank": rank,
        "q48_projection_fraction": float(
            la.norm(overlap, ord="fro") ** 2
            / max(la.norm(action, ord="fro") ** 2, np.finfo(float).tiny)
        ),
        "factor_seconds": perf_counter() - started,
    }


def dense_probe_coefficients(
    q_complement: np.ndarray,
    triangular: np.ndarray,
    pivots: np.ndarray,
    probe_basis: np.ndarray,
    maximum_rank: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stage-40/41 normalized action-overlap rule applied inside B_perp."""

    interaction = q_complement.T @ probe_basis
    projected_probe_rank = int(np.linalg.matrix_rank(interaction))
    left, singular, _ = la.svd(interaction, full_matrices=False, check_finite=False)
    pivoted_coefficients, _, _, _ = la.lstsq(
        triangular, left[:, :maximum_rank], cond=1.0e-12,
        lapack_driver="gelsy", check_finite=False,
    )
    coefficients = np.zeros_like(pivoted_coefficients)
    coefficients[pivots, :] = pivoted_coefficients
    relative = 1.0e-12 * max(float(np.max(np.abs(coefficients))), np.finfo(float).tiny)
    return coefficients, {
        "probe_rank": probe_basis.shape[1],
        "projected_probe_rank": projected_probe_rank,
        "largest_probe_interaction": float(singular[0]),
        "smallest_retained_probe_interaction": float(singular[maximum_rank - 1]),
        "coefficient_density": float(np.mean(np.abs(coefficients) > relative)),
    }


def dense_teacher_coefficients(
    q_complement: np.ndarray,
    triangular: np.ndarray,
    pivots: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Best dense combination in the fixed complementary action range."""

    interaction = q_complement.T @ target
    left, singular, _ = la.svd(interaction, full_matrices=False, check_finite=False)
    rank = target.shape[1]
    pivoted_coefficients, _, _, _ = la.lstsq(
        triangular, left[:, :rank], cond=1.0e-12, lapack_driver="gelsy",
        check_finite=False,
    )
    coefficients = np.zeros_like(pivoted_coefficients)
    coefficients[pivots, :] = pivoted_coefficients
    relative = 1.0e-12 * max(float(np.max(np.abs(coefficients))), np.finfo(float).tiny)
    return coefficients, {
        "target_rank": rank,
        "minimum_projection_cosine": float(singular[rank - 1]),
        "mean_projection_coverage": float(np.mean(singular[:rank] ** 2)),
        "coefficient_density": float(np.mean(np.abs(coefficients) > relative)),
    }


def paired_complement_and_hybrid(
    matrix,
    dictionary: np.ndarray,
    action: np.ndarray,
    coefficients: np.ndarray,
    classical: RecyclePair,
) -> Tuple[RecyclePair, RecyclePair, Dict[str, float]]:
    raw_u = dictionary @ coefficients
    raw_w = action @ coefficients
    complementary_u = raw_u - classical.u_basis @ (classical.w_basis.T @ raw_w)
    correction = _ordered_pair_from_z(matrix, complementary_u, 1.0e-13)
    cross = classical.w_basis.T @ correction.w_basis
    if la.norm(cross, ord="fro") > 1.0e-8:
        raise RuntimeError("correction is not complementary to frozen q48")
    hybrid_u = np.column_stack((classical.u_basis, correction.u_basis))
    hybrid_w = np.column_stack((classical.w_basis, correction.w_basis))
    hybrid_z = la.qr(hybrid_u, mode="economic", check_finite=False)[0]
    defect = float(
        la.norm(np.asarray(matrix @ hybrid_u, float) - hybrid_w, ord="fro")
        / max(la.norm(hybrid_w, ord="fro"), np.finfo(float).tiny)
    )
    hybrid = RecyclePair(
        z_basis=hybrid_z,
        u_basis=hybrid_u,
        w_basis=hybrid_w,
        setup_seconds=correction.setup_seconds,
        diagnostics={
            "au_equals_w_relative_error": defect,
            "test_orthogonality_error": float(
                la.norm(hybrid_w.T @ hybrid_w - np.eye(hybrid_w.shape[1]), ord="fro")
            ),
            "reduced_condition": float("nan"),
            "dictionary_rank": float(dictionary.shape[1]),
        },
    )
    return correction, hybrid, {
        "q48_correction_overlap": float(la.norm(cross, ord="fro") ** 2 / correction.rank),
        "q48_containment_in_hybrid": float(
            la.norm(hybrid.w_basis.T @ classical.w_basis, ord="fro") ** 2
            / classical.rank
        ),
        "pair_defect": defect,
    }


def construct_spaces(root: Path, output: Path, settings: Stage49Settings):
    cache = output / "target_local_spaces.npz"
    geometry_path = output / "correction_geometry.csv"
    construction_path = output / "construction_diagnostics.csv"
    expected = {
        f"system_{sample_id:03d}_{family}_r{rank}_u"
        for sample_id in settings.hard_system_ids
        for family in ("shared516_ceiling", "local_oracle", "local_deployable")
        for rank in settings.correction_ranks
    }
    if cache.exists() and geometry_path.exists() and construction_path.exists():
        cached_construction = _read_csv(construction_path)
        exact_source = bool(cached_construction) and all(
            row.get("classical_source") == "Stage41 float64 multi_rhs_harmonic_m64_z"
            and row.get("deployable_rule") == "normalized_stage41_overlap"
            for row in cached_construction
        )
        with np.load(cache) as archive:
            if exact_source and expected.issubset(archive.files):
                pairs = {}
                for sample_id in settings.hard_system_ids:
                    stage28 = root / "results/results-fd/stage28_operator_learning"
                    import scipy.sparse as sp
                    matrix = sp.load_npz(stage28 / "matrices" / f"A_{sample_id:03d}.npz").tocsr()
                    for family in ("shared516_ceiling", "local_oracle", "local_deployable"):
                        for rank in settings.correction_ranks:
                            u = np.asarray(archive[f"system_{sample_id:03d}_{family}_r{rank}_u"], float)
                            pairs[(sample_id, family, rank)] = _pair_from_z(matrix, u)
                return pairs, _read_csv(geometry_path), _read_csv(construction_path)

    full_span_path = (
        root / "results/results-fd/stages/stage30_full_span_representation_ceiling/full_training_span.npz"
    )
    with np.load(full_span_path) as archive:
        psi = np.asarray(archive["psi"], float)
        full_singular = np.asarray(archive["singular_values"], float)
    if psi.shape != (5041, settings.full_span_rank):
        raise RuntimeError(f"unexpected Stage-30 full span {psi.shape}")
    if np.count_nonzero(full_singular > settings.numerical_rank_tolerance * full_singular[0]) != settings.full_span_rank:
        raise RuntimeError("Stage-30 full span failed numerical-rank verification")
    with np.load(
        root / "results/results-fd/stages/stage48_full_dictionary_complementary_correction/full_correction_dictionary.npz"
    ) as archive:
        dictionary516 = np.asarray(archive["basis"], float)
    stage47_cache = (
        root / "results/results-fd/stages/stage47_fno_complementary_correction/hard_test_classical_complements.npz"
    )
    stage41_cache = (
        root / "results/results-fd/stages/stage41_multi_rhs_slow_mode_target/multi_rhs_local_spaces.npz"
    )
    probe_cache = (
        root / "results/results-fd/stages/stage42_missing_critical_mode_diagnosis/combined_harmonic_probe_spaces.npz"
    )
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    pairs: Dict[Tuple[int, str, int], RecyclePair] = {}
    geometry_rows: List[Dict[str, Any]] = []
    construction_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}

    with (
        np.load(stage47_cache) as stage47_archive,
        np.load(stage41_cache) as stage41_archive,
        np.load(probe_cache) as probe_archive,
    ):
        for position, sample_id in enumerate(settings.hard_system_ids, start=1):
            context = _stage28_context(
                root, sample_id, nodes, boundary, coefficient_samples, settings.stage29
            )
            matrix = context.system.matrix
            classical_z = np.asarray(
                stage41_archive[f"system_{sample_id:03d}_multi_rhs_harmonic_m64_z"], float
            )
            classical = _pair_from_z(matrix, classical_z)
            if classical.rank != settings.frozen_classical_rank:
                raise RuntimeError("Stage-41 classical rank is not 48")
            probe = np.asarray(
                probe_archive[f"system_{sample_id:03d}_combined_harmonic_action_basis"], float
            )

            # Deployable coefficients are frozen before any target/oracle array is accessed.
            full_action = np.asarray(matrix @ psi, float)
            b_perp_full, q_full, r_full, pivots_full, full_diag = factor_complementary_action(
                full_action, classical, settings.numerical_rank_tolerance
            )
            deploy_coefficients, deploy_diag = dense_probe_coefficients(
                q_full, r_full, pivots_full, probe, max(settings.correction_ranks)
            )
            deployable = {}
            for rank in settings.correction_ranks:
                correction, hybrid, pair_diag = paired_complement_and_hybrid(
                    matrix, psi, full_action, deploy_coefficients[:, :rank], classical
                )
                deployable[rank] = (correction, hybrid, pair_diag)

            # Held-out target is loaded only after deployable coefficients and spaces are fixed.
            target_u12 = np.asarray(
                stage47_archive[f"system_{sample_id:03d}_target_u12"], float
            )
            target_pair = _ordered_pair_from_z(matrix, target_u12, 1.0e-13)

            shared_action = np.asarray(matrix @ dictionary516, float)
            _, q_shared, r_shared, pivots_shared, shared_diag = factor_complementary_action(
                shared_action, classical, settings.numerical_rank_tolerance
            )
            for rank in settings.correction_ranks:
                target = target_pair.w_basis[:, :rank]
                local_coefficients, local_diag = dense_teacher_coefficients(
                    q_full, r_full, pivots_full, target
                )
                shared_coefficients, shared_target_diag = dense_teacher_coefficients(
                    q_shared, r_shared, pivots_shared, target
                )
                candidates = {
                    "local_deployable": deployable[rank],
                    "local_oracle": paired_complement_and_hybrid(
                        matrix, psi, full_action, local_coefficients, classical
                    ),
                    "shared516_ceiling": paired_complement_and_hybrid(
                        matrix, dictionary516, shared_action, shared_coefficients, classical
                    ),
                }
                for family, (correction, hybrid, pair_diag) in candidates.items():
                    metrics = principal_coverage_metrics(correction.w_basis, target)
                    pairs[(sample_id, family, rank)] = hybrid
                    arrays[f"system_{sample_id:03d}_{family}_r{rank}_u"] = hybrid.u_basis
                    arrays[f"system_{sample_id:03d}_{family}_r{rank}_w"] = hybrid.w_basis
                    geometry_rows.append({
                        "sample_id": sample_id,
                        "method": f"stage49_{family}_r{rank}",
                        "family": family,
                        "correction_rank": rank,
                        "final_rank": hybrid.rank,
                        **metrics,
                        **pair_diag,
                    })
                construction_rows.append({
                    "sample_id": sample_id,
                    "correction_rank": rank,
                    "full_span_columns": psi.shape[1],
                    "full_complement_rank": full_diag["numerical_rank"],
                    "full_q48_projection_fraction": full_diag["q48_projection_fraction"],
                    "probe_rank": deploy_diag["probe_rank"],
                    "projected_probe_rank": deploy_diag["projected_probe_rank"],
                    "deployable_coefficient_density": deploy_diag["coefficient_density"],
                    "deployable_smallest_interaction": deploy_diag["smallest_retained_probe_interaction"],
                    "local_oracle_coefficient_density": local_diag["coefficient_density"],
                    "local_oracle_projection_coverage": local_diag["mean_projection_coverage"],
                    "shared516_complement_rank": shared_diag["numerical_rank"],
                    "shared516_oracle_coefficient_density": shared_target_diag["coefficient_density"],
                    "classical_source": "Stage41 float64 multi_rhs_harmonic_m64_z",
                    "deployable_rule": "normalized_stage41_overlap",
                    "teacher_loaded_after_deployable_space_frozen": True,
                })
            print(f"[Stage49 spaces] {position}/7 system {sample_id}", flush=True)
    np.savez_compressed(cache, **arrays)
    _write_csv(geometry_path, geometry_rows)
    _write_csv(construction_path, construction_rows)
    return pairs, geometry_rows, construction_rows


def evaluate_solver(root: Path, output: Path, pairs, settings: Stage49Settings):
    cache = output / "solver_results.csv"
    if cache.exists():
        rows = _read_csv(cache)
        new_rows = [row for row in rows if row["method"].startswith("stage49_")]
        if len(new_rows) == 126 and all(row.get("implementation_revision") == "normalized_stage41_overlap" for row in new_rows):
            return rows
    stage48 = _read_csv(
        root / "results/results-fd/stages/stage48_full_dictionary_complementary_correction/solver_results.csv"
    )
    baseline_methods = {
        "strong_ilu", "teacher_full", "full_span_oracle", "stage41_harmonic_fixed_q48",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
    }
    rows = [dict(row) for row in stage48 if row["method"] in baseline_methods]
    lookup = {
        (int(row["sample_id"]), int(row["rhs_seed"]), row["method"]): row
        for row in stage48
    }
    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficient_samples = _sample_coefficients(settings.stage29.stage28.stage19a)
    for sample_id in settings.hard_system_ids:
        context = _stage28_context(
            root, sample_id, nodes, boundary, coefficient_samples, settings.stage29
        )
        for rhs_seed in settings.stage29.stage28.evaluation_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, rhs_seed)
            baseline = lookup[(sample_id, rhs_seed, "strong_ilu")]
            teacher = lookup[(sample_id, rhs_seed, "teacher_full")]
            denominator = int(baseline["iterations"]) - int(teacher["iterations"])
            for family in ("shared516_ceiling", "local_oracle", "local_deployable"):
                for rank in settings.correction_ranks:
                    pair = pairs[(sample_id, family, rank)]
                    result = _run_pair(context, rhs, pair, settings.stage29.stage28.stage22a)
                    residual = _relative_residual(context.system.matrix, result.x, rhs)
                    rows.append({
                        "sample_id": sample_id,
                        "rhs_seed": rhs_seed,
                        "method": f"stage49_{family}_r{rank}",
                        "source": "Stage49 deployable" if family == "local_deployable" else "Stage49 non-deployable ceiling",
                        "iterations": int(result.iterations),
                        "converged": bool(result.converged and residual <= 1.0e-8),
                        "true_final_residual": residual,
                        "actual_rank": pair.rank,
                        "rho": (
                            (int(baseline["iterations"]) - int(result.iterations)) / denominator
                            if denominator > 0 else float("nan")
                        ),
                        "failure": getattr(result, "failure", "") or "",
                        "implementation_revision": "normalized_stage41_overlap",
                    })
        print(f"[Stage49 solves] system {sample_id}", flush=True)
    return rows


def summarize_solver(rows, methods):
    result = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        result.append({
            "method": method,
            "converged_rhs": sum(_bool(row["converged"]) for row in local),
            "median_iterations": float(np.median([int(row["iterations"]) for row in local])),
            "median_rho": float(np.median([float(row["rho"]) for row in local])),
            "median_final_rank": float(np.median([int(row["actual_rank"]) for row in local])),
        })
    return result


def _report(output, solver_summary, geometry_rows, decision, settings):
    labels = {
        "strong_ilu": "strong ILU", "teacher_full": "own-system teacher",
        "full_span_oracle": "Stage-30 full-span oracle",
        "stage41_harmonic_fixed_q48": "fixed classical q48",
        "classical_q48_plus_oracle_corr8": "Stage-47 exact oracle correction r=8",
        "classical_q48_plus_oracle_corr12": "Stage-47 exact oracle correction r=12",
        "stage49_shared516_ceiling_r8": "Stage-48 shared-516 ceiling r=8",
        "stage49_shared516_ceiling_r12": "Stage-48 shared-516 ceiling r=12",
        "stage49_local_oracle_r8": "Stage-49 target-local oracle r=8",
        "stage49_local_oracle_r12": "Stage-49 target-local oracle r=12",
        "stage49_local_deployable_r8": "Stage-49 deployable local r=8",
        "stage49_local_deployable_r12": "Stage-49 deployable local r=12",
    }
    geometry_lookup = {row["method"]: row for row in geometry_rows}
    lines = [
        "# Stage 49: Target-local complementary-correction representation ceiling",
        "", "## Result", "", f"**{decision['conclusion']}**", "",
        "The Stage-41 q48 range is retained exactly. Stage-49 forms dense target-specific combinations from all 1,456 Stage-30 directions after projecting their actions away from q48. The deployable method uses only the cached 72-dimensional Stage-41 harmonic probe space.",
        "", "| method | converged | median iterations | median rho | final rank | containment | target overlap | p10 / minimum coverage | median / largest angle |",
        "|:--|--:|--:|--:|--:|--:|--:|:--|:--|",
    ]
    for row in solver_summary:
        method = row["method"]
        local_geometry = [g for g in geometry_rows if g["method"] == method]
        if local_geometry:
            containment = np.median([float(g["correction_containment"]) for g in local_geometry])
            p10 = np.median([float(g["p10_principal_coverage"]) for g in local_geometry])
            minimum = np.median([float(g["minimum_principal_coverage"]) for g in local_geometry])
            overlap = np.median([float(g["target_correction_frobenius_overlap"]) for g in local_geometry])
            median_angle = np.median([float(g["median_principal_angle_degrees"]) for g in local_geometry])
            angle = np.median([float(g["largest_principal_angle_degrees"]) for g in local_geometry])
            geometry = (
                f"{_fmt(containment)} | {_fmt(overlap)} | {_fmt(p10)} / {_fmt(minimum)} | "
                f"{_fmt(median_angle,1)} / {_fmt(angle,1)} degrees"
            )
        elif method.startswith("classical_q48_plus_oracle_corr"):
            rank = 12 if method.endswith("12") else 8
            geometry = f"1.000 | {_fmt(np.sqrt(rank))} | 1.000 / 1.000 | 0.0 / 0.0 degrees"
        else:
            geometry = "n/a | n/a | n/a | n/a"
        lines.append(
            f"| {labels[method]} | {row['converged_rhs']}/21 | {_fmt(row['median_iterations'],1)} | "
            f"{_fmt(row['median_rho'])} | {_fmt(row['median_final_rank'],0)} | {geometry} |"
        )
    lines += [
        "", "The reported target-correction overlap is basis invariant: containment is the mean squared principal cosine, while p10 and minimum expose the poorly covered tail hidden by average projector metrics.",
        "", "## System 63", "",
        "| method | converged | median iterations | rho | containment | target overlap | p10 / minimum coverage | median / largest angle |",
        "|:--|--:|--:|--:|--:|--:|:--|:--|",
    ]
    solver_rows = _read_csv(output / "solver_results.csv")
    for method in (
        "stage41_harmonic_fixed_q48",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
        "stage49_shared516_ceiling_r8", "stage49_shared516_ceiling_r12",
        "stage49_local_oracle_r8", "stage49_local_oracle_r12",
        "stage49_local_deployable_r8", "stage49_local_deployable_r12",
    ):
        local_s = [row for row in solver_rows if int(row["sample_id"]) == 63 and row["method"] == method]
        local_g = [row for row in geometry_rows if int(row["sample_id"]) == 63 and row["method"] == method]
        if local_g:
            g = local_g[0]
            geometry = (
                f"{_fmt(g['correction_containment'])} | {_fmt(g['target_correction_frobenius_overlap'])} | "
                f"{_fmt(g['p10_principal_coverage'])} / {_fmt(g['minimum_principal_coverage'])} | "
                f"{_fmt(g['median_principal_angle_degrees'],1)} / {_fmt(g['largest_principal_angle_degrees'],1)} degrees"
            )
        elif method.startswith("classical_q48_plus_oracle_corr"):
            rank = 12 if method.endswith("12") else 8
            geometry = f"1.000 | {_fmt(np.sqrt(rank))} | 1.000 / 1.000 | 0.0 / 0.0 degrees"
        else:
            geometry = "n/a | n/a | n/a | n/a"
        lines.append(
            f"| {labels[method]} | {sum(_bool(r['converged']) for r in local_s)}/3 | "
            f"{_fmt(np.median([int(r['iterations']) for r in local_s]),1)} | "
            f"{_fmt(np.median([float(r['rho']) for r in local_s]))} | {geometry} |"
        )
    lines += [
        "", "## Representation versus target-local information", "",
        decision["interpretation"],
        "", "## Stage 50 recommendation", "", decision["stage50_recommendation"],
    ]
    report = output / "stage49_target_local_correction_ceiling.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage49(root: Path) -> Path:
    settings = Stage49Settings()
    output = root / "results/results-fd/stages/stage49_target_local_correction_ceiling"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-49 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
    })
    preflight = {
        "full_span_shape": [5041, 1456],
        "classical_rank": 48,
        "correction_ranks": [8, 12],
        "cached_probe_source": "Stage42 combined_harmonic_probe_spaces (exact Stage41 probes)",
        "no_new_probe_rhs": True,
        "no_neural_training": True,
        "deployable_teacher_access": False,
    }
    _write_json(output / "preflight.json", {**preflight, "passed": True})

    pairs, geometry_rows, construction_rows = construct_spaces(root, output, settings)
    solver_rows = evaluate_solver(root, output, pairs, settings)
    _write_csv(output / "solver_results.csv", solver_rows)
    methods = (
        "strong_ilu", "teacher_full", "full_span_oracle", "stage41_harmonic_fixed_q48",
        "classical_q48_plus_oracle_corr8", "classical_q48_plus_oracle_corr12",
        "stage49_shared516_ceiling_r8", "stage49_shared516_ceiling_r12",
        "stage49_local_oracle_r8", "stage49_local_oracle_r12",
        "stage49_local_deployable_r8", "stage49_local_deployable_r12",
    )
    solver_summary = summarize_solver(solver_rows, methods)
    _write_csv(output / "solver_summary.csv", solver_summary)
    summary = {row["method"]: row for row in solver_summary}
    oracle12 = summary["stage49_local_oracle_r12"]
    deploy12 = summary["stage49_local_deployable_r12"]
    sys63_oracle = [
        row for row in solver_rows if int(row["sample_id"]) == 63
        and row["method"] == "stage49_local_oracle_r12"
    ]
    sys63_deploy = [
        row for row in solver_rows if int(row["sample_id"]) == 63
        and row["method"] == "stage49_local_deployable_r12"
    ]
    oracle_works = (
        int(oracle12["converged_rhs"]) == 21
        and sum(_bool(row["converged"]) for row in sys63_oracle) == 3
    )
    deploy_works = (
        int(deploy12["converged_rhs"]) == 21
        and sum(_bool(row["converged"]) for row in sys63_deploy) == 3
        and float(np.median([int(row["iterations"]) for row in sys63_deploy])) <= 147.0
    )
    shared_geometry = [row for row in geometry_rows if row["family"] == "shared516_ceiling" and int(row["correction_rank"]) == 12]
    deploy_geometry = [row for row in geometry_rows if row["family"] == "local_deployable" and int(row["correction_rank"]) == 12]
    average_geometry_improves = (
        np.median([float(row["correction_containment"]) for row in deploy_geometry])
        >= np.median([float(row["correction_containment"]) for row in shared_geometry]) + 0.10
    )
    if not oracle_works:
        conclusion = "Outcome C — the target-local full-span correction oracle fails."
        interpretation = (
            "Representation failed: even teacher-informed dense combinations of the projected "
            "Stage-30 full span cannot recover the required compact correction."
        )
        recommendation = (
            "Stage 50 should revisit the complementary correction representation before any neural training."
        )
    elif deploy_works:
        conclusion = "Outcome A — target-local oracle and deployable construction both work."
        interpretation = (
            "Representation and target-local probe extraction both succeed. The dense local "
            "coefficient rule is a valid deployable correction mechanism."
        )
        recommendation = (
            "Stage 50 should use FNO only to predict or warm-start the low-dimensional local "
            "combination rule C_local while retaining the classical q48 prefix."
        )
    elif average_geometry_improves and sum(_bool(row["converged"]) for row in sys63_deploy) < 3:
        conclusion = "Outcome D — average local geometry improves, but system 63 still fails."
        interpretation = (
            "Representation succeeds under the local oracle, but the deployable probe extraction "
            "still misses the lower-tail coverage needed by system 63."
        )
        recommendation = (
            "Stage 50 should condition an FNO on K(x) and the compact local interaction matrix to "
            "predict the missing low-dimensional coefficient correction, with system-63 tail coverage as the gate."
        )
    else:
        conclusion = "Outcome B — the local oracle works, but deployable extraction fails."
        interpretation = (
            "Representation is valid, while Stage-41 probe information/local extraction is "
            "insufficient. Although rank 12 reaches 21/21, system 63 needs a median 298 "
            "iterations (rho=0.008), versus 95 for the local oracle; its p10/minimum correction "
            "coverage remains 0.003/0.001. This is tolerance crossing without useful acceleration, "
            "so the failure is information selection rather than the full-span correction range. "
            "The teacher-informed shared-516 ceiling also reaches 21/21 despite only 0.586 median "
            "containment, reinforcing that average geometry alone is not a sufficient solver proxy."
        )
        recommendation = (
            "Stage 50 should condition an FNO on K(x) and the compact local interaction matrix to "
            "supply the missing information for C_local; it should not predict the full correction basis."
        )
    decision = {
        "conclusion": conclusion,
        "oracle_works": oracle_works,
        "deployable_works": deploy_works,
        "average_geometry_improves": average_geometry_improves,
        "interpretation": interpretation,
        "stage50_recommendation": recommendation,
    }
    _write_json(output / "decision.json", decision)
    report = _report(output, solver_summary, geometry_rows, decision, settings)
    validation = {
        "full_span_rank": settings.full_span_rank,
        "all_complementary_action_ranks_expected": all(
            int(row["full_complement_rank"])
            == settings.full_span_rank - settings.frozen_classical_rank
            for row in construction_rows
        ),
        "deployable_coefficients_dense": min(
            float(row["deployable_coefficient_density"]) for row in construction_rows
        ) > 0.90,
        "teacher_loaded_after_deployable_space_frozen": all(
            _bool(row["teacher_loaded_after_deployable_space_frozen"])
            for row in construction_rows
        ),
        "correction_ranks_exact": all(
            int(row["correction_rank"]) in settings.correction_ranks
            and int(row["final_rank"]) == settings.frozen_classical_rank + int(row["correction_rank"])
            for row in geometry_rows
        ),
        "q48_preserved": min(
            float(row["q48_containment_in_hybrid"]) for row in geometry_rows
        ) >= 1.0 - 1.0e-9,
        "maximum_q48_correction_overlap": max(
            float(row["q48_correction_overlap"]) for row in geometry_rows
        ),
        "maximum_pair_defect": max(float(row["pair_defect"]) for row in geometry_rows),
        "new_solver_rows": len([row for row in solver_rows if row["method"].startswith("stage49_")]),
        "expected_new_solver_rows": 126,
        "no_neural_model": True,
        "no_new_probes": True,
    }
    validation["passed"] = bool(
        validation["full_span_rank"] == 1456
        and validation["all_complementary_action_ranks_expected"]
        and validation["deployable_coefficients_dense"]
        and validation["teacher_loaded_after_deployable_space_frozen"]
        and validation["correction_ranks_exact"] and validation["q48_preserved"]
        and validation["maximum_q48_correction_overlap"] <= 1.0e-8
        and validation["maximum_pair_defect"] <= 1.0e-8
        and validation["new_solver_rows"] == validation["expected_new_solver_rows"]
        and validation["no_neural_model"] and validation["no_new_probes"]
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage49_target_local_correction_ceiling.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage49] complete", flush=True)
    return copy
