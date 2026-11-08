"""Stage 50: attribute observable information for the unresolved correction tail."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as la

from ..subspace_analysis import generalized_subspace_metrics
from .stage19a_spatial_coefficient_oracle import _discrete_rhs, _sample_coefficients
from .stage19c_oracle_mechanism import _capture_arnoldi
from .stage29_family_level_coarse_space import _stage28_context
from .stage30_full_span_representation_ceiling import _fmt, _manifest, _read_csv
from .stage41_multi_rhs_slow_mode_target import Stage41Settings, harmonic_action_information
from .stage42_missing_critical_mode_diagnosis import _orthonormal_basis
from .stage49_target_local_correction_ceiling import principal_coverage_metrics
from .stage9_scalability import _write_csv, _write_json


@dataclass(frozen=True)
class Stage50Settings:
    hard_system_ids: Tuple[int, ...] = (13, 25, 40, 43, 55, 63, 71)
    prefix_depths: Tuple[int, ...] = (8, 16, 32, 64)
    tail_rank_tolerance: float = 1.0e-11
    observable_singular_tolerance: float = 1.0e-3
    early_coverage_gate: float = 0.25
    early_growth_gate: float = 0.10
    material_variation_gate: float = 0.50

    @property
    def stage41(self) -> Stage41Settings:
        return Stage41Settings()

    @property
    def stage29(self):
        return self.stage41.stage29


def unresolved_tail(
    deployable_correction: np.ndarray,
    oracle_correction: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Basis-invariant range of (I-P_deployable) W_oracle."""

    deployable = _orthonormal_basis(deployable_correction)
    oracle = _orthonormal_basis(oracle_correction)
    residual = oracle - deployable @ (deployable.T @ oracle)
    left, singular, _ = la.svd(residual, full_matrices=False, check_finite=False)
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    if rank == 0:
        raise RuntimeError("Stage-49 deployable correction leaves no unresolved tail")
    tail = left[:, :rank]
    return tail, {
        "deployable_rank": deployable.shape[1],
        "oracle_rank": oracle.shape[1],
        "tail_rank": rank,
        "largest_tail_singular_value": float(singular[0]),
        "smallest_retained_tail_singular_value": float(singular[rank - 1]),
        "tail_oracle_containment": principal_coverage_metrics(tail, oracle)["correction_containment"],
    }


def coverage_row(
    sample_id: int,
    information: str,
    prefix: int,
    rhs_seed: int | str,
    space: np.ndarray,
    tail: np.ndarray,
) -> Dict[str, Any]:
    metrics = principal_coverage_metrics(space, tail)
    return {
        "sample_id": sample_id,
        "information": information,
        "prefix": prefix,
        "rhs_seed": rhs_seed,
        "tail_rank": tail.shape[1],
        **metrics,
    }


def observable_tail_space(
    information: np.ndarray,
    tail: np.ndarray,
    singular_tolerance: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    basis = _orthonormal_basis(information)
    left, singular, _ = la.svd(basis.T @ tail, full_matrices=False, check_finite=False)
    rank = int(np.count_nonzero(singular > singular_tolerance))
    if rank == 0:
        return np.empty((tail.shape[0], 0)), {
            "observable_rank": 0,
            "largest_observable_cosine": float(singular[0]),
            "smallest_retained_cosine": float("nan"),
        }
    return basis @ left[:, :rank], {
        "observable_rank": rank,
        "largest_observable_cosine": float(singular[0]),
        "smallest_retained_cosine": float(singular[rank - 1]),
    }


def pairwise_geometry(first: np.ndarray, second: np.ndarray) -> Dict[str, float]:
    if first.shape[1] == 0 or second.shape[1] == 0:
        return {
            "subspace_distance": 1.0,
            "mutual_containment": 0.0,
            "median_principal_angle_degrees": 90.0,
            "largest_principal_angle_degrees": 90.0,
        }
    metrics = generalized_subspace_metrics(first, second)
    angles = np.asarray(metrics["principal_angles_degrees"], float)
    return {
        "subspace_distance": float(metrics["projector_distance_normalized"]),
        "mutual_containment": float(
            min(metrics["containment_first"], metrics["containment_second"])
        ),
        "median_principal_angle_degrees": float(np.median(angles)),
        "largest_principal_angle_degrees": float(np.max(angles)),
    }


def build_tail_targets(root: Path, output: Path, settings: Stage50Settings):
    path = root / "results/results-fd/stages/stage49_target_local_correction_ceiling/target_local_spaces.npz"
    tails, rows, arrays = {}, [], {}
    with np.load(path) as archive:
        for sample_id in settings.hard_system_ids:
            deployable = np.asarray(
                archive[f"system_{sample_id:03d}_local_deployable_r12_w"], float
            )[:, -12:]
            oracle = np.asarray(
                archive[f"system_{sample_id:03d}_local_oracle_r12_w"], float
            )[:, -12:]
            tail, diagnostics = unresolved_tail(
                deployable, oracle, settings.tail_rank_tolerance
            )
            tails[sample_id] = tail
            arrays[f"system_{sample_id:03d}_tail_w"] = tail
            rows.append({"sample_id": sample_id, **diagnostics})
    np.savez_compressed(output / "unresolved_tail_spaces.npz", **arrays)
    _write_csv(output / "tail_target_diagnostics.csv", rows)
    return tails, rows


def reconstruct_cached_protocol_states(
    root: Path, output: Path, tails: Mapping[int, np.ndarray], settings: Stage50Settings
):
    cache = output / "stage41_diagnostic_states.npz"
    provenance_path = output / "probe_reconstruction.csv"
    expected = {
        f"system_{sample_id:03d}_seed_{seed}_raw64"
        for sample_id in settings.hard_system_ids for seed in settings.stage41.probe_rhs_seeds
    } | {
        f"system_{sample_id:03d}_seed_{seed}_harmonic24"
        for sample_id in settings.hard_system_ids for seed in settings.stage41.probe_rhs_seeds
    }
    if cache.exists() and provenance_path.exists():
        with np.load(cache) as archive:
            current_format = (
                "cache_format_version" in archive.files
                and int(np.asarray(archive["cache_format_version"]).item()) == 2
            )
            if current_format and expected.issubset(archive.files):
                states = {
                    (sample_id, seed): {
                        "raw": np.asarray(archive[f"system_{sample_id:03d}_seed_{seed}_raw64"], float),
                        "harmonic": np.asarray(archive[f"system_{sample_id:03d}_seed_{seed}_harmonic24"], float),
                    }
                    for sample_id in settings.hard_system_ids
                    for seed in settings.stage41.probe_rhs_seeds
                }
                return states, _read_csv(provenance_path)

    fixed = np.load(
        root / "results/results-fd/stages/stage19A_spatial_coefficient_oracle_validation/fixed_nodes.npz"
    )
    nodes = np.asarray(fixed["nodes"], float)
    boundary = np.asarray(fixed["boundary"], bool)
    coefficients = _sample_coefficients(settings.stage29.stage28.stage19a)
    combined_cache = np.load(
        root / "results/results-fd/stages/stage42_missing_critical_mode_diagnosis/combined_harmonic_probe_spaces.npz"
    )
    arrays, states, rows = {}, {}, []
    for position, sample_id in enumerate(settings.hard_system_ids, start=1):
        context = _stage28_context(root, sample_id, nodes, boundary, coefficients, settings.stage29)
        local_harmonic = []
        for seed in settings.stage41.probe_rhs_seeds:
            rhs, _ = _discrete_rhs(context.system, seed)
            started = perf_counter()
            arnoldi = _capture_arnoldi(
                context.system.matrix, context.ilu, rhs, settings.stage41.probe_depth
            )
            harmonic_action, _, _, _ = harmonic_action_information(
                context.system.matrix, arnoldi, settings.stage41.slow_modes_per_probe
            )
            raw = np.asarray(
                arnoldi.residual_basis[:, : settings.stage41.probe_depth], float
            )
            # Preserve the exact action columns. Stage 42 formed the combined
            # space from these columns before its single global QR.
            harmonic = np.asarray(harmonic_action, float)
            arrays[f"system_{sample_id:03d}_seed_{seed}_raw64"] = raw
            arrays[f"system_{sample_id:03d}_seed_{seed}_harmonic24"] = harmonic
            states[(sample_id, seed)] = {"raw": raw, "harmonic": harmonic}
            local_harmonic.append(harmonic)
            rows.append({
                "sample_id": sample_id,
                "rhs_seed": seed,
                "requested_depth": settings.stage41.probe_depth,
                "actual_depth": int(arnoldi.hessenberg.shape[1]),
                "harmonic_rank": int(harmonic_action.shape[1]),
                "seconds": perf_counter() - started,
                "existing_stage41_rhs_reconstructed": True,
            })
        reconstructed = _orthonormal_basis(np.column_stack(local_harmonic))
        cached = np.asarray(
            combined_cache[f"system_{sample_id:03d}_combined_harmonic_action_basis"], float
        )
        comparison = generalized_subspace_metrics(reconstructed, cached)
        rows[-1]["combined_cached_projector_distance"] = comparison["projector_distance_normalized"]
        rows[-1]["combined_cached_containment"] = comparison["containment_second"]
        print(f"[Stage50 probes] {position}/7 system {sample_id}", flush=True)
    combined_cache.close()
    arrays["cache_format_version"] = np.asarray(2)
    np.savez_compressed(cache, **arrays)
    _write_csv(provenance_path, rows)
    return states, rows


def analyze_information(
    tails: Mapping[int, np.ndarray],
    states: Mapping[Tuple[int, int], Mapping[str, np.ndarray]],
    settings: Stage50Settings,
):
    coverage_rows: List[Dict[str, Any]] = []
    observed: Dict[Tuple[int, int, str], np.ndarray] = {}
    observable_rows: List[Dict[str, Any]] = []
    for sample_id in settings.hard_system_ids:
        tail = tails[sample_id]
        for seed in settings.stage41.probe_rhs_seeds:
            state = states[(sample_id, seed)]
            for depth in settings.prefix_depths:
                raw = _orthonormal_basis(state["raw"][:, :depth])
                coverage_rows.append(
                    coverage_row(sample_id, "raw_individual", depth, seed, raw, tail)
                )
            harmonic = state["harmonic"]
            coverage_rows.append(
                coverage_row(sample_id, "harmonic_individual", 24, seed, harmonic, tail)
            )
            for information, space in (("raw64", state["raw"]), ("harmonic", harmonic)):
                local, diagnostics = observable_tail_space(
                    space, tail, settings.observable_singular_tolerance
                )
                observed[(sample_id, seed, information)] = local
                observable_rows.append({
                    "sample_id": sample_id,
                    "rhs_seed": seed,
                    "information": information,
                    **diagnostics,
                })
        for depth in settings.prefix_depths:
            combined = _orthonormal_basis(np.column_stack([
                states[(sample_id, seed)]["raw"][:, :depth]
                for seed in settings.stage41.probe_rhs_seeds
            ]))
            coverage_rows.append(
                coverage_row(sample_id, "raw_combined", depth, "combined", combined, tail)
            )
        combined_harmonic = _orthonormal_basis(np.column_stack([
            states[(sample_id, seed)]["harmonic"] for seed in settings.stage41.probe_rhs_seeds
        ]))
        coverage_rows.append(
            coverage_row(
                sample_id, "harmonic_combined", 24, "combined", combined_harmonic, tail
            )
        )

    same_k_rows = []
    substitutability_rows = []
    for sample_id in settings.hard_system_ids:
        for information in ("raw64", "harmonic"):
            for first_seed, second_seed in combinations(settings.stage41.probe_rhs_seeds, 2):
                first = observed[(sample_id, first_seed, information)]
                second = observed[(sample_id, second_seed, information)]
                same_k_rows.append({
                    "sample_id": sample_id,
                    "information": information,
                    "first_rhs_seed": first_seed,
                    "second_rhs_seed": second_seed,
                    **pairwise_geometry(first, second),
                })
            for source_seed in settings.stage41.probe_rhs_seeds:
                source = states[(sample_id, source_seed)]["raw" if information == "raw64" else "harmonic"]
                for target_seed in settings.stage41.probe_rhs_seeds:
                    if source_seed == target_seed:
                        continue
                    target_observed = observed[(sample_id, target_seed, information)]
                    if target_observed.shape[1] == 0:
                        containment = 0.0
                    else:
                        containment = principal_coverage_metrics(
                            source, target_observed
                        )["correction_containment"]
                    substitutability_rows.append({
                        "sample_id": sample_id,
                        "information": information,
                        "source_rhs_seed": source_seed,
                        "target_rhs_seed": target_seed,
                        "information_substitutability": containment,
                        "solver_cross_rhs_not_run": True,
                        "reason": "no cached per-RHS correction; constructing one is forbidden in Stage 50",
                    })

    different_k_rows = []
    for information in ("raw64", "harmonic"):
        for seed in settings.stage41.probe_rhs_seeds:
            for first_system, second_system in combinations(settings.hard_system_ids, 2):
                first = observed[(first_system, seed, information)]
                second = observed[(second_system, seed, information)]
                different_k_rows.append({
                    "information": information,
                    "rhs_seed": seed,
                    "first_system": first_system,
                    "second_system": second_system,
                    **pairwise_geometry(first, second),
                })
    target_k_rows = []
    for first_system, second_system in combinations(settings.hard_system_ids, 2):
        target_k_rows.append({
            "first_system": first_system,
            "second_system": second_system,
            **pairwise_geometry(tails[first_system], tails[second_system]),
        })
    return (
        coverage_rows, observable_rows, same_k_rows,
        substitutability_rows, different_k_rows, target_k_rows,
    )


def median_metric(rows, **filters):
    local = [
        row for row in rows
        if all(str(row[key]) == str(value) for key, value in filters.items())
    ]
    return local


def build_summaries(
    coverage_rows, same_k_rows, substitutability_rows, different_k_rows,
    target_k_rows, settings: Stage50Settings,
):
    coverage_summary = []
    for information, depths in (
        ("raw_individual", settings.prefix_depths),
        ("raw_combined", settings.prefix_depths),
        ("harmonic_individual", (24,)),
        ("harmonic_combined", (24,)),
    ):
        for depth in depths:
            local = [
                row for row in coverage_rows
                if row["information"] == information and int(row["prefix"]) == depth
            ]
            coverage_summary.append({
                "information": information,
                "prefix": depth,
                "observations": len(local),
                "median_containment": float(np.median([row["correction_containment"] for row in local])),
                "median_p10_coverage": float(np.median([row["p10_principal_coverage"] for row in local])),
                "median_minimum_coverage": float(np.median([row["minimum_principal_coverage"] for row in local])),
            })
    variation_summary = []
    for information in ("raw64", "harmonic"):
        within = [row for row in same_k_rows if row["information"] == information]
        across = [row for row in different_k_rows if row["information"] == information]
        substitute = [row for row in substitutability_rows if row["information"] == information]
        variation_summary.append({
            "information": information,
            "D_RHS": float(np.median([row["subspace_distance"] for row in within])),
            "D_K": float(np.median([row["subspace_distance"] for row in across])),
            "median_same_K_mutual_containment": float(np.median([row["mutual_containment"] for row in within])),
            "median_different_K_mutual_containment": float(np.median([row["mutual_containment"] for row in across])),
            "median_cross_RHS_information_substitutability": float(np.median([row["information_substitutability"] for row in substitute])),
        })
    variation_summary.append({
        "information": "target_tail_across_K",
        "D_RHS": float("nan"),
        "D_K": float(np.median([row["subspace_distance"] for row in target_k_rows])),
        "median_same_K_mutual_containment": float("nan"),
        "median_different_K_mutual_containment": float(np.median([row["mutual_containment"] for row in target_k_rows])),
        "median_cross_RHS_information_substitutability": float("nan"),
    })
    return coverage_summary, variation_summary


def decide(coverage_summary, variation_summary, settings: Stage50Settings):
    coverage = {(row["information"], int(row["prefix"])): row for row in coverage_summary}
    variation = {row["information"]: row for row in variation_summary}
    raw8 = coverage[("raw_combined", 8)]["median_containment"]
    raw64 = coverage[("raw_combined", 64)]["median_containment"]
    raw_gain = raw64 - raw8
    d_rhs = variation["raw64"]["D_RHS"]
    d_k = variation["raw64"]["D_K"]
    harmonic = coverage[("harmonic_combined", 24)]["median_containment"]
    early_informative = raw64 >= settings.early_coverage_gate and raw_gain >= settings.early_growth_gate
    rhs_material = d_rhs >= settings.material_variation_gate
    k_material = d_k >= settings.material_variation_gate
    if k_material and d_k >= 1.5 * max(d_rhs, 1.0e-12) and not early_informative:
        outcome = "Outcome A — K dominates."
        neural_input = "K"
        interpretation = (
            "Same-K RHS variation is small relative to different-K variation, while richer raw "
            "Krylov prefixes add little tail coverage. K(x) is the primary missing information."
        )
    elif rhs_material and early_informative and not k_material:
        outcome = "Outcome B — early Krylov state dominates."
        neural_input = "early Krylov"
        interpretation = (
            "Same-K RHS variation is material and raw-prefix coverage grows substantially; early "
            "Krylov state is essential, with K optional."
        )
    elif k_material and rhs_material and early_informative:
        outcome = "Outcome C — both K and early Krylov state matter."
        neural_input = "K + early Krylov"
        interpretation = (
            "Changing K and changing the construction RHS both materially alter the observable "
            "tail, and progressively richer raw Krylov prefixes expose additional tail content."
        )
    else:
        outcome = "Outcome D — neither observable source adequately explains the tail."
        neural_input = "neither"
        interpretation = (
            "The unresolved tail remains weakly observable from raw prefixes without a stable "
            "same-K/different-K attribution. Current observable information is insufficient."
        )
    return {
        "conclusion": outcome,
        "recommended_stage51_neural_input": neural_input,
        "raw_combined_m8_containment": raw8,
        "raw_combined_m64_containment": raw64,
        "raw_prefix_containment_gain": raw_gain,
        "D_RHS_raw64": d_rhs,
        "D_K_raw64": d_k,
        "harmonic_combined_containment": harmonic,
        "raw64_minus_harmonic_containment": raw64 - harmonic,
        "harmonic_compression_discards_observable_tail": raw64 > harmonic + 0.05,
        "early_state_informative": early_informative,
        "RHS_variation_material": rhs_material,
        "K_variation_material": k_material,
        "interpretation": interpretation,
    }


def _report(
    output, coverage_rows, coverage_summary, variation_summary, same_k_rows,
    substitutability_rows, tail_rows, decision, settings,
):
    lines = [
        "# Stage 50: Information attribution for the missing complementary tail",
        "", "## Result", "", f"**{decision['conclusion']}**", "",
        decision["interpretation"],
        "", "The diagnostic target is the basis-invariant range of `(I-P_Wdep)W_oracle`. No recycle space or solver method was constructed. Raw states were reconstructed with the exact three Stage-41 RHS seeds and depth 64 because those vectors were not archived; reconstructed harmonic spans were checked against the Stage-42 cache.",
        "", "Stage 49 materialized paired deployable/oracle corrections only for the seven hard systems. The earlier training/validation split remains frozen, but Stage 50 does not synthesize missing train/validation corrections because that would create a new correction construction prohibited by this stage.",
        "", "## Raw-prefix and harmonic observability", "",
        "| information | prefix | observations | median containment | median p10 | median minimum |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for row in coverage_summary:
        lines.append(
            f"| {row['information']} | {row['prefix']} | {row['observations']} | "
            f"{_fmt(row['median_containment'])} | {_fmt(row['median_p10_coverage'])} | "
            f"{_fmt(row['median_minimum_coverage'])} |"
        )
    lines += [
        "", "## Same-K RHS variation versus different-K variation", "",
        "| information | D_RHS | D_K | same-K containment | different-K containment | cross-RHS substitutability |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for row in variation_summary:
        lines.append(
            f"| {row['information']} | {_fmt(row['D_RHS'])} | {_fmt(row['D_K'])} | "
            f"{_fmt(row['median_same_K_mutual_containment'])} | "
            f"{_fmt(row['median_different_K_mutual_containment'])} | "
            f"{_fmt(row['median_cross_RHS_information_substitutability'])} |"
        )
    lines += [
        "", "Cross-RHS substitutability is an information-space containment test. Solver substitution was not run because no per-RHS correction spaces were cached, and constructing them would violate Stage 50's non-repetition constraint.",
        "", "The Stage-49 tail target is fixed for each K. Same-K distances therefore quantify how the observable projection/signature of that fixed target changes with construction RHS; they do not claim that separate RHS-specific oracle targets were cached.",
        "", "## System 63", "",
        "| RHS | raw m8 | raw m16 | raw m32 | raw m64 | harmonic |",
        "|--:|--:|--:|--:|--:|--:|",
    ]
    for seed in settings.stage41.probe_rhs_seeds:
        values = {}
        for depth in settings.prefix_depths:
            row = next(
                row for row in coverage_rows
                if int(row["sample_id"]) == 63 and row["information"] == "raw_individual"
                and int(row["prefix"]) == depth and int(row["rhs_seed"]) == seed
            )
            values[depth] = row
        harmonic = next(
            row for row in coverage_rows
            if int(row["sample_id"]) == 63 and row["information"] == "harmonic_individual"
            and int(row["rhs_seed"]) == seed
        )
        lines.append(
            f"| {seed} | " + " | ".join(
                f"{_fmt(values[depth]['correction_containment'])} ({_fmt(values[depth]['p10_principal_coverage'])}/{_fmt(values[depth]['minimum_principal_coverage'])})"
                for depth in settings.prefix_depths
            ) + f" | {_fmt(harmonic['correction_containment'])} ({_fmt(harmonic['p10_principal_coverage'])}/{_fmt(harmonic['minimum_principal_coverage'])}) |"
        )
    combined63 = [
        row for row in coverage_rows
        if int(row["sample_id"]) == 63 and row["information"] in {"raw_combined", "harmonic_combined"}
    ]
    same_k_63 = {
        information: float(np.median([
            row["subspace_distance"] for row in same_k_rows
            if int(row["sample_id"]) == 63 and row["information"] == information
        ]))
        for information in ("raw64", "harmonic")
    }
    same_k_ordinary = {
        information: float(np.median([
            row["subspace_distance"] for row in same_k_rows
            if int(row["sample_id"]) != 63 and row["information"] == information
        ]))
        for information in ("raw64", "harmonic")
    }
    tail_rank_63 = next(
        int(row["tail_rank"]) for row in tail_rows if int(row["sample_id"]) == 63
    )
    tail_rank_median = int(np.median([int(row["tail_rank"]) for row in tail_rows]))
    lines += [
        "", "Values are containment with `(p10/minimum)` in parentheses. Combined system-63 coverage: "
        + "; ".join(
            f"{row['information']} m={row['prefix']}: {_fmt(row['correction_containment'])} ({_fmt(row['p10_principal_coverage'])}/{_fmt(row['minimum_principal_coverage'])})"
            for row in combined63
        ) + ".",
        "", f"The unresolved target has median rank {tail_rank_median}; system 63 has rank {tail_rank_63}. "
        f"Its same-K cross-RHS distance is {_fmt(same_k_63['raw64'])} for raw m64 and "
        f"{_fmt(same_k_63['harmonic'])} after harmonic filtering, compared with ordinary-system "
        f"medians {_fmt(same_k_ordinary['raw64'])} and {_fmt(same_k_ordinary['harmonic'])}.",
        "", "Across the six ordinary hard systems, raw-combined m64 median containment is "
        + _fmt(np.median([
            row["correction_containment"] for row in coverage_rows
            if row["information"] == "raw_combined" and int(row["prefix"]) == 64
            and int(row["sample_id"]) != 63
        ])) + ", versus " + _fmt(next(
            row["correction_containment"] for row in combined63
            if row["information"] == "raw_combined" and int(row["prefix"]) == 64
        )) + " for system 63.",
        "", "Raw-versus-harmonic diagnostic: raw-combined m64 median containment minus combined-harmonic "
        f"containment is {_fmt(decision['raw64_minus_harmonic_containment'])}. "
        + ("The fixed harmonic compression discards observable tail content."
           if decision["harmonic_compression_discards_observable_tail"]
           else "The fixed harmonic compression is not the main source of lost observable tail content."),
        "", "## Recommended Stage-51 neural input", "",
        f"**{decision['recommended_stage51_neural_input']}**",
    ]
    report = output / "stage50_missing_tail_information_attribution.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_stage50(root: Path) -> Path:
    settings = Stage50Settings()
    output = root / "results/results-fd/stages/stage50_missing_tail_information_attribution"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "predeclared_protocol.json").exists():
        raise RuntimeError("Stage-50 protocol missing")
    _write_json(output / "settings.json", asdict(settings))
    _write_json(output / "environment.json", {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
    })
    tails, tail_rows = build_tail_targets(root, output, settings)
    states, reconstruction_rows = reconstruct_cached_protocol_states(
        root, output, tails, settings
    )
    (
        coverage_rows, observable_rows, same_k_rows,
        substitutability_rows, different_k_rows, target_k_rows,
    ) = analyze_information(tails, states, settings)
    coverage_summary, variation_summary = build_summaries(
        coverage_rows, same_k_rows, substitutability_rows,
        different_k_rows, target_k_rows, settings,
    )
    decision = decide(coverage_summary, variation_summary, settings)
    _write_csv(output / "tail_coverage.csv", coverage_rows)
    _write_csv(output / "observable_tail_ranks.csv", observable_rows)
    _write_csv(output / "same_K_rhs_variation.csv", same_k_rows)
    _write_csv(output / "cross_rhs_substitutability.csv", substitutability_rows)
    _write_csv(output / "different_K_variation.csv", different_k_rows)
    _write_csv(output / "target_tail_different_K_variation.csv", target_k_rows)
    _write_csv(output / "coverage_summary.csv", coverage_summary)
    _write_csv(output / "variation_summary.csv", variation_summary)
    _write_json(output / "decision.json", decision)
    report = _report(
        output, coverage_rows, coverage_summary, variation_summary, same_k_rows,
        substitutability_rows, tail_rows, decision, settings,
    )
    validation = {
        "no_neural_model": True,
        "no_new_correction_construction": True,
        "probe_seeds_exact": list(settings.stage41.probe_rhs_seeds),
        "probe_depth_exact": settings.stage41.probe_depth,
        "prefixes_exact": list(settings.prefix_depths),
        "all_tail_ranks_positive": all(int(row["tail_rank"]) > 0 for row in tail_rows),
        "all_reconstructed_depths_64": all(
            int(row["actual_depth"]) == 64 for row in reconstruction_rows
        ) if reconstruction_rows else True,
        "maximum_cached_harmonic_projector_distance": max(
            [float(row["combined_cached_projector_distance"])
             for row in reconstruction_rows if row.get("combined_cached_projector_distance", "") != ""]
            or [0.0]
        ),
        "coverage_rows": len(coverage_rows),
        "expected_coverage_rows": 7 * (3 * 4 + 3 + 4 + 1),
        "teacher_used_only_for_diagnostic_tail": True,
        "cross_rhs_solver_runs": 0,
    }
    validation["passed"] = bool(
        validation["no_neural_model"] and validation["no_new_correction_construction"]
        and validation["probe_seeds_exact"] == list(Stage41Settings().probe_rhs_seeds)
        and validation["probe_depth_exact"] == 64
        and validation["prefixes_exact"] == [8, 16, 32, 64]
        and validation["all_tail_ranks_positive"]
        and validation["all_reconstructed_depths_64"]
        and validation["maximum_cached_harmonic_projector_distance"] <= 2.0e-8
        and validation["coverage_rows"] == validation["expected_coverage_rows"]
        and validation["teacher_used_only_for_diagnostic_tail"]
        and validation["cross_rhs_solver_runs"] == 0
    )
    _write_json(output / "validation.json", validation)
    _write_json(output / "manifest.json", _manifest(output))
    copy = root / "results/results-fd/reports/stage50_missing_tail_information_attribution.md"
    copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print("[Stage50] complete", flush=True)
    return copy
