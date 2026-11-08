#!/usr/bin/env python3
"""Finalize the Stage-20 evidence report from saved numerical outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from lcp_rbffd.experiments.stage20_spatial_k_robust_oracle import (
    BLUE,
    CHARCOAL,
    GRAY,
    LIGHT_GRAY,
    ORANGE,
    Stage20Settings,
    _write_manifest,
)
from lcp_rbffd.experiments.stage9_scalability import _write_csv, _write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def median(values) -> float:
    return float(statistics.median(values))


def truth(value: str) -> bool:
    return value.lower() == "true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(root: Path) -> None:
    settings = Stage20Settings()
    fd_root = root / "results" / "results-fd"
    output = fd_root / "stages" / "stage20_spatial_K_robust_oracle_search"
    report = fd_root / "reports" / "stage20_spatial_K_robust_oracle_search.md"
    stage19a = fd_root / "stages" / "stage19A_spatial_coefficient_oracle_validation"
    stage19c = fd_root / "stages" / "stage19C_spatial_coefficient_oracle_mechanism_diagnostic"

    reproduction = read_csv(output / "reproduction_checks.csv")
    diagnostics = read_csv(output / "nonnormality_diagnostics.csv")
    old_stability = read_csv(output / "construction_subspace_stability.csv")
    spectrum = read_csv(output / "long_arnoldi_spectrum.csv")
    blocks = read_csv(output / "spectral_block_ablation.csv")
    # Remove a pre-finalization bookkeeping defect: a block outside the q=8
    # prefix had been recorded as a deletion even though the basis was unchanged.
    blocks = [
        row
        for row in blocks
        if not (row.get("ablation") == "delete_from_q8" and int(row["q"]) == 8)
    ]
    _write_csv(output / "spectral_block_ablation.csv", blocks)
    selection = read_csv(output / "candidate_selection_results.csv")
    summaries = read_csv(output / "heldout_oracle_summary.csv")
    family = read_csv(output / "family_gate_summary.csv")
    rank = read_csv(output / "arnoldi_rank_ablation.csv")
    stability = read_csv(output / "m128_construction_stability.csv")
    hardness = {
        int(row["sample_id"]): row
        for row in read_csv(stage19a / "hardness_scan_summary.csv")
    }
    baseline = read_csv(stage19c / "baseline_results.csv")

    ids = settings.sample_ids
    q_requested = settings.q_values

    def rank_rows(sample_id: int, method: str) -> list[dict[str, str]]:
        return [
            row
            for row in rank
            if int(row["sample_id"]) == sample_id and row["method"] == method
        ]

    def rank_stat(sample_id: int, family_name: str, q: int) -> tuple[int, int, float, bool]:
        local = rank_rows(sample_id, f"{family_name}_m128_q{q}")
        actual = int(local[0]["q"])
        k = int(median(int(row["iterations"]) for row in local))
        k_ilu = int(median(int(row["k_ilu"]) for row in local))
        gamma = (k_ilu - k) / k_ilu
        converged = all(truth(row["converged"]) for row in local)
        return actual, k, gamma, converged

    family_success_by_q = {}
    for family_name in ("ordinary", "harmonic"):
        for q in q_requested:
            family_success_by_q[(family_name, q)] = sum(
                rank_stat(sample_id, family_name, q)[3]
                and rank_stat(sample_id, family_name, q)[2] >= settings.strong_gamma
                for sample_id in ids
            )

    # Clear rank-response plot with independent system traces.
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(ids)))
    for axis, family_name in zip(axes, ("ordinary", "harmonic")):
        for color, sample_id in zip(colors, ids):
            actual = []
            iterations = []
            for q in q_requested:
                q_actual, k, _, _ = rank_stat(sample_id, family_name, q)
                actual.append(q_actual)
                iterations.append(k)
            axis.plot(actual, iterations, marker="o", markersize=3.5, color=color, label=f"ID {sample_id}")
        axis.axhline(settings.solver.max_iterations, color=ORANGE, linestyle="--", linewidth=1)
        axis.set_title(f"{family_name.capitalize()} Ritz")
        axis.set_xlabel("Actual pair-respecting recycle rank")
        axis.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0].set_ylabel("Median held-out iterations")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle("Long-Arnoldi rank sensitivity ($m=128$)")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "plots" / "rank_sensitivity_m128.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Geometry-versus-function plot for independent q≈24 constructions.
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    annotation_offsets = {
        3: (5, 6),
        7: (5, 6),
        10: (5, 6),
        14: (-18, -13),
        15: (5, 14),
        17: (5, -13),
        19: (5, 8),
        34: (5, -1),
    }
    for sample_id in ids:
        local = [
            row
            for row in stability
            if int(row["sample_id"]) == sample_id
            and row["method"] == "ordinary_m128_q24_construction_stability"
            and int(row["construction_rhs_seed"]) != settings.construction_rhs_seed
        ]
        by_seed = defaultdict(list)
        for row in local:
            by_seed[int(row["construction_rhs_seed"])].append(row)
        distances = [float(group[0]["projector_distance_normalized"]) for group in by_seed.values()]
        gammas = [median(float(value["gamma"]) for value in group) for group in by_seed.values()]
        distance = median(distances)
        gamma = median(gammas)
        passed = min(gammas) >= settings.strong_gamma
        axis.scatter(
            distance,
            gamma,
            color=BLUE if passed else ORANGE,
            edgecolor=CHARCOAL,
            linewidth=0.4,
            s=48,
        )
        axis.annotate(
            str(sample_id),
            (distance, gamma),
            xytext=annotation_offsets[sample_id],
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(settings.strong_gamma, color=ORANGE, linestyle="--", label="40% solver gate")
    axis.set_xlabel("Normalized projector distance from primary construction")
    axis.set_ylabel("Median held-out iteration reduction")
    axis.set_title(r"Geometric motion does not imply loss of solver action ($q\approx24$)")
    axis.grid(color=LIGHT_GRAY, linewidth=0.6)
    figure.tight_layout()
    figure.savefig(output / "plots" / "construction_stability_vs_solver.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Re-render the archived-history diagnostic with a non-overlapping title
    # and legend.  Numerical values come directly from the saved history CSV.
    histories = read_csv(output / "residual_histories.csv")
    figure, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, ids):
        for method, color, linestyle in (
            ("strong_ilu", CHARCOAL, "-"),
            ("old_q8_ordinary", ORANGE, "--"),
        ):
            local = [
                row
                for row in histories
                if int(row["sample_id"]) == sample_id and row["method"] == method
            ]
            axis.semilogy(
                [int(row["iteration"]) for row in local],
                [float(row["relative_residual"]) for row in local],
                color=color,
                linestyle=linestyle,
                label=method.replace("_", " "),
            )
        axis.set_title(f"ID {sample_id}")
        axis.grid(color=LIGHT_GRAY, linewidth=0.6)
    axes[0, 0].set_ylabel("True relative residual")
    axes[1, 0].set_ylabel("True relative residual")
    for axis in axes[1, :]:
        axis.set_xlabel("Iteration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Baseline and archived $q=8$ ordinary-Ritz residual histories", y=0.995)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output / "plots" / "old_q8_residual_histories.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    def reproduction_median(sample_id: int, definition: str) -> int:
        return int(
            median(
                int(row["reproduced_iterations"])
                for row in reproduction
                if int(row["sample_id"]) == sample_id and row["definition"] == definition
            )
        )

    lines = [
        "# Stage 20: robust complementary-space search for spatially varying anisotropy",
        "",
        "## Decision",
        "",
        "**A repeatable own-system oracle has been recovered on a majority of the hard spatial-$K(x)$ systems, but the recovery comes from a longer Arnoldi construction and a moderate block of directions, not from isolating a small fixed spectral cluster.** With the unchanged strong ILU, the archived 64-step, $q=8$ ordinary-Ritz oracle succeeded only for ID 15. Extending construction to 128 Arnoldi steps gives substantial, true-residual-converged acceleration on 6 of 8 hard systems by requested $q=24$; pair preservation makes the actual ranks 23 or 24. Ordinary and harmonic Ritz give the same family success count. Increasing to $q=64$ reduces median iterations further but does not rescue IDs 10 or 17 under the primary construction.",
        "",
        r"The predeclared gate required all three held-out RHSs to converge and at least 40% median iteration reduction on at least 5 of the 8 systems. The $m=128$, $q\approx24$ ordinary space passes 6/8 and remains a 6/8 success under two independent construction RHSs. Its dimension is only 0.46--0.48% of $N=5041$, so the experiment supports a low-to-moderate-dimensional solver action for most, but not all, of this hard family. It does **not** yet establish a stable geometric target for supervised learning: several functionally useful spaces have maximum principal angles near $90^\circ$.",
        "",
        "No neural model was trained. Because the spectral family gate passed, the predeclared conditional multilevel/local-global fallback was not run.",
        "",
        "## Frozen problem and oracle protocol",
        "",
        "The eight systems are Stage-19A IDs 3, 7, 10, 14, 15, 17, 19, and 34. The archived sparse matrices are reused exactly. Each has $N=5041$, stencil size 25, PHS $r^3$ with degree-2 polynomial augmentation, the fixed seed-7 jittered node set with jitter 0.15, and the same smooth spatially varying anisotropic tensor $K(x)$ recorded in `coefficient_samples.csv`. SuperLU ILU uses drop tolerance $10^{-3}$, fill factor 5, COLAMD ordering, and the established zero diagonal pivot threshold. GMRES/recycled GMRES uses right preconditioning, restart 50, relative tolerance $10^{-8}$, and a cap of 300 iterations.",
        "",
        "Arnoldi is applied to $B=AM^{-1}$. For a residual-space Ritz vector $y_i$, the corresponding solution-side direction is $z_i=M^{-1}y_i$. Complete real directions and complex-conjugate pairs are retained together and orthonormalized to form $Z$. The unchanged nonsymmetric recycling pair is constructed by thin QR, $AZ=WR$, followed by triangular solves for $U=ZR^{-1}$, so $AU=W$ and $W^T W=I$. Recycled GMRES applies Arnoldi to $(I-WW^T)AM^{-1}$ and always stops on the true residual of the original system.",
        "",
        f"Construction used RHS seed {settings.construction_rhs_seed}. Adaptive candidate selection used seeds {settings.selection_rhs_seeds[0]} and {settings.selection_rhs_seeds[1]}. Final evaluation used the untouched Stage-19A seeds {settings.heldout_rhs_seeds[0]}--{settings.heldout_rhs_seeds[-1]}. The selection and evaluation trajectories are disjoint. The additional rank table is an ex-post oracle dimensionality diagnostic, not a deployed rank selector.",
        "",
        "## Archived failure reproduced",
        "",
        "Every archived strong-ILU and $q=8$ ordinary-Ritz, harmonic-Ritz, and correction-POD iteration count was rerun from the saved matrices and matched exactly for all three held-out RHSs. The cap is a censored failure, not a converged estimate.",
        "",
        "| ID | strong ILU | ordinary Ritz $q=8$ | harmonic Ritz $q=8$ | correction POD $q=8$ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for sample_id in ids:
        lines.append(
            f"| {sample_id} | {reproduction_median(sample_id, 'strong_ilu')} | {reproduction_median(sample_id, 'ordinary_ritz')} | {reproduction_median(sample_id, 'harmonic_ritz')} | {reproduction_median(sample_id, 'correction_pod')} |"
        )
    lines.extend(
        [
            "",
            "![Archived q=8 residual histories](../stages/stage20_spatial_K_robust_oracle_search/plots/old_q8_residual_histories.png)",
            "",
            "Correction POD preserves the original missing-action definition $d(r)=A^{-1}r-M^{-1}r$, with $A^{-1}r$ computed by accurate solves rather than an explicit inverse. Its failure here is solver-level: the $q=8$ POD space reaches 77 iterations for ID 15 and the cap for the other seven systems. Stage 19C also found only 1/8 strong successes through $q=32$, even when snapshot energy was highly concentrated. Compression energy therefore remains distinct from Krylov relevance.",
            "",
            "## Why ID 15 was the exceptional $q=8$ case",
            "",
            r"The evidence supports a construction-convergence explanation. With the original 64-step Arnoldi process, ID 15 has a maximum ordinary $q=8$ Ritz-pair residual of $2.65\times10^{-5}$, while the other systems range from 0.302 to 0.864. Its independently constructed ordinary $q=8$ spaces are also identical to reported precision, whereas the other systems have median normalized projector distances from 0.446 to 0.676 and median largest angles from $64.0^\circ$ to $86.1^\circ$.",
            "",
            "| ID | max ordinary $q=8$ pair residual, $m=64$ | median $q=8$ projector distance | median largest angle | projected commutator | Henrici departure | projected eigenvector cond. |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics:
        sample_id = int(row["sample_id"])
        local = [
            value
            for value in old_stability
            if int(value["sample_id"]) == sample_id
            and value["family"] == "ordinary_m64"
            and int(value["q"]) == 8
        ]
        lines.append(
            rf"| {sample_id} | {float(row['ordinary_q8_residual_max']):.3g} | {median(float(value['projector_distance_normalized']) for value in local):.3f} | {median(float(value['principal_angle_max_degrees']) for value in local):.1f}$^\circ$ | {float(row['projected_commutator_normalized']):.3f} | {float(row['projected_henrici_departure']):.3f} | {float(row['projected_eigenvector_condition']):.0f} |"
        )
    lines.extend(
        [
            "",
            "The longer construction changes the spectral approximation itself. The following ranges use the first eight pair-respecting directions from the 128-step run.",
            "",
            r"| ID | ordinary $|\lambda|$ range | max ordinary pair residual | harmonic $|\lambda|$ range | max harmonic pair residual |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for sample_id in ids:
        values = {}
        for family_name in ("ordinary", "harmonic"):
            local = [
                row
                for row in spectrum
                if int(row["sample_id"]) == sample_id
                and row["family"] == family_name
                and int(row["direction"]) <= 8
            ]
            magnitudes = [float(row["magnitude"]) for row in local]
            residuals = [float(row["ritz_residual_relative"]) for row in local]
            values[family_name] = (min(magnitudes), max(magnitudes), max(residuals))
        lines.append(
            f"| {sample_id} | {values['ordinary'][0]:.3g}--{values['ordinary'][1]:.3g} | {values['ordinary'][2]:.3g} | {values['harmonic'][0]:.3g}--{values['harmonic'][1]:.3g} | {values['harmonic'][2]:.3g} |"
        )
    lines.extend(
        [
            "",
            "ID 15 is not the least nonnormal system. Its projected commutator (1.38) and Henrici departure (0.991) are comparable with IDs 3, 17, 19, and 34. Small Ritz magnitudes also do not by themselves explain success. In the 128-step construction, the first eight ordinary Ritz magnitudes span 0.0063--0.1197 for ID 15, while all systems contain values near zero. What distinguishes ID 15 at short construction is the residual quality and construction-RHS repeatability of the selected pairs.",
            "",
            "Harmonic extraction lowers the maximum first-eight pair residual for several cases, most notably ID 17 (0.684 to 0.154), but both $q=8$ spaces still fail there. Ordinary and harmonic rank-response curves become nearly identical once enough converged directions are included. Better individual pair residuals are therefore helpful diagnostics, but insufficient without adequate aggregate rank.",
            "",
            "On selection RHS 20261031, ID 15's first ordinary atomic Ritz block alone reduces 238 ILU iterations to 128, and the full $q=8$ space reduces the same solve to 41. No individual ordinary or harmonic block among the first 16 directions gives a converged improvement for any other hard ID. Spectral-gap clusters do not expose a small universal block either: the only effective cluster used alone is an ID-15 cluster of rank 54. The solver-relevant information for the newly rescued systems is therefore distributed over several blocks.",
            "",
            "![Ritz quality and nonnormality diagnostics](../stages/stage20_spatial_K_robust_oracle_search/plots/id15_mechanism_diagnostics.png)",
            "",
            "## What actually repairs the oracle",
            "",
            "At fixed rank 32, extending Arnoldi from 64 to 128 steps changes both ordinary and harmonic family success from 2/8 to 6/8. At fixed $m=128$, the ordinary-Ritz success count is 3/8 at requested $q=8$, 5/8 at $q=16$, and 6/8 at $q=24$; it remains 6/8 at 32, 48, and 64. Harmonic Ritz has the same counts. This identifies Arnoldi convergence as the largest discrete design change and moderate rank as the second. The family result is not attributable to a different oracle target or solver integration.",
            "",
            "| ID | ILU | $m=128$, $q=8$ | $q=16$ | $q=24$ | $q=32$ | $q=64$ | smallest passing actual $q$ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample_id in ids:
        stats = [rank_stat(sample_id, "ordinary", q) for q in q_requested]
        q_min = next((actual for actual, _, gamma, converged in stats if converged and gamma >= settings.strong_gamma), None)
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                sample_id,
                reproduction_median(sample_id, "strong_ilu"),
                *[f"{value[1]} ($q={value[0]}$)" for value in (stats[0], stats[1], stats[2], stats[3], stats[5])],
                "--" if q_min is None else q_min,
            )
        )
    lines.extend(
        [
            "",
            "![Rank sensitivity](../stages/stage20_spatial_K_robust_oracle_search/plots/rank_sensitivity_m128.png)",
            "",
            "The pair-aware actual ranks can be one below the requested value. The first family-wide passing requested rank, $q=24$, is 0.48% of $N$; actual rank 23 is 0.46%. ID 10 converges but reaches only 19.7% reduction at $q=23$ and 39.0% at $q=64$, just below the gate. ID 17 remains at the cap for every tested rank. These are explicit failures, not omitted outliers.",
            "",
            "Residual-quality filters and spectral-gap cluster combinations do not improve the gate. The selection rule chooses $q=64$ for every adaptive-prefix system because the objective minimizes iterations, and the filtered and cluster-combination variants reproduce nearly the same $q=64$ spaces and iteration counts. Thus the useful ablation is longer Arnoldi plus sufficient aggregate rank. Filtering and clustering add search cost without family-wide solver value.",
            "",
            "## Own-system gate, true residuals, and costs",
            "",
            "The next table uses the simpler ordinary $m=128$, requested-$q=24$ construction. Setup is the measured Arnoldi/Ritz construction plus the nonsymmetric $AZ=WR$ pairing. Online time is the median recycled solve. ILU setup is included for context and is not weakened or omitted.",
            "",
            "| ID | actual $q$ | ILU median | recycle median [min,max] | reduction | converged | ILU setup (s) | recycle setup (s) | online (s) | gate |",
            "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|",
        ]
    )
    for sample_id in ids:
        local = rank_rows(sample_id, "ordinary_m128_q24")
        actual = int(local[0]["q"])
        kvals = [int(row["iterations"]) for row in local]
        k = int(median(kvals))
        k_ilu = int(median(int(row["k_ilu"]) for row in local))
        gamma = (k_ilu - k) / k_ilu
        converged = all(truth(row["converged"]) for row in local)
        setup = float(local[0]["construction_seconds"]) + float(local[0]["pair_setup_seconds"])
        online = median(float(row["online_seconds"]) for row in local)
        passed = converged and gamma >= settings.strong_gamma
        lines.append(
            f"| {sample_id} | {actual} | {k_ilu} | {k} [{min(kvals)},{max(kvals)}] | {gamma:.3f} | {'yes' if converged else 'no'} | {float(hardness[sample_id]['ilu_setup_seconds']):.3f} | {setup:.3f} | {online:.3f} | {'pass' if passed else 'fail'} |"
        )
    lines.extend(
        [
            "",
            r"The $q\approx24$ gate passes IDs 3, 7, 14, 15, 19, and 34. All reported passes have maximum true residual at or below the $10^{-8}$ stopping criterion. The recycle construction costs roughly an order of magnitude more than one ILU factorization in this $N=5041$ diagnostic, so it only has an economic role when amortized across repeated solves. The adaptive candidate search is much more expensive: median search times for the $q=8$--64 prefix family are about 2.0--8.9 seconds per system and residual-filter sweeps can exceed 40 seconds. Those search costs are oracle diagnostics and are not hidden in online time.",
            "",
            "## Construction stability and suitability as a later target",
            "",
            r"The same $m=128$ ordinary spaces were reconstructed independently from RHS seeds 20261001, 20261011, and 20261021. At $q\approx24$, all three constructions pass on the same six systems and fail on IDs 10 and 17. Functional stability is therefore strong enough for the own-system oracle conclusion. Geometric stability is weaker.",
            "",
            "| ID | median iterations across construction RHSs | alternate projector distance range | alternate largest-angle range | functionally robust gate |",
            "|---:|---:|---:|---:|:---:|",
        ]
    )
    for sample_id in ids:
        local = [
            row
            for row in stability
            if int(row["sample_id"]) == sample_id
            and row["method"] == "ordinary_m128_q24_construction_stability"
        ]
        by_seed = defaultdict(list)
        for row in local:
            by_seed[int(row["construction_rhs_seed"])].append(row)
        k_values = [int(median(int(row["iterations"]) for row in group)) for group in by_seed.values()]
        alternate = [row for row in local if int(row["construction_rhs_seed"]) != settings.construction_rhs_seed]
        distances = [float(row["projector_distance_normalized"]) for row in alternate]
        angles = [float(row["principal_angle_max_degrees"]) for row in alternate]
        robust = all(
            all(truth(row["converged"]) for row in group)
            and median(float(row["gamma"]) for row in group) >= settings.strong_gamma
            for group in by_seed.values()
        )
        lines.append(
            rf"| {sample_id} | {min(k_values)}--{max(k_values)} | {min(distances):.3f}--{max(distances):.3f} | {min(angles):.1f}$^\circ$--{max(angles):.1f}$^\circ$ | {'yes' if robust else 'no'} |"
        )
    lines.extend(
        [
            "",
            "![Construction stability versus solver response](../stages/stage20_spatial_K_robust_oracle_search/plots/construction_stability_vs_solver.png)",
            "",
            r"For IDs 3 and 7, normalized projector distances of 0.32--0.40 and largest angles above $80^\circ$ coexist with strong solver acceleration. Conversely, ID 17 remains ineffective even as its space changes. Geometric reproducibility is therefore not the right own-system success criterion. A future predictor would need to preserve the recycle action relevant to projected Krylov convergence, not reproduce a particular Ritz basis entrywise. This repeats the earlier finding that projector geometry and solver value can disagree.",
            "",
            "The result is still not a neural-learning gate. The construction depends on 128 applications of $AM^{-1}$ and own-system Krylov information. Cross-field predictability and a cheaper label must be demonstrated separately. The current evidence supports a stable *solver effect* on six systems, not a smoothly parameterized geometric subspace.",
            "",
            "## Limitations",
            "",
            "The family contains eight preselected hard fields from Stage 19A. Each result uses three held-out RHSs and a cap of 300, so differences among cap hits are unidentifiable. Ranks above 64 were not tested. The adaptive search is intentionally system-own and expensive. Nonnormality indicators are projected or randomized matrix-free proxies, not dense exact norms. The multilevel/local-global branch was not run because the predeclared spectral majority gate passed; there is therefore no local-space claim in this report.",
            "",
            "## One next step",
            "",
            "**Run one out-of-sample oracle validation of the fixed $m=128$, requested-$q=24$ ordinary-Ritz construction on the remaining numerically valid Stage-19A spatial fields, using fresh construction-RHS groups and the same untouched strong ILU/recycled-GMRES protocol.** This single test should estimate the family-wide success rate and functional construction stability before any neural prediction is reconsidered.",
            "",
            "## Reproducibility and provenance",
            "",
            "Primary evidence is saved under `results/results-fd/stages/stage20_spatial_K_robust_oracle_search/`:",
            "",
            "- `reproduction_checks.csv`: exact archived-result reruns.",
            "- `nonnormality_diagnostics.csv`, `construction_subspace_stability.csv`, and `long_arnoldi_spectrum.csv`: ID-15 mechanism diagnostics.",
            "- `spectral_block_ablation.csv` and `candidate_selection_results.csv`: individual, cluster, filter, and selection ablations.",
            "- `arnoldi_rank_ablation.csv`: held-out fixed-depth/fixed-rank comparison.",
            "- `m128_construction_stability.csv`: independent-construction geometry and solver response.",
            "- `heldout_oracle_results.csv`, `heldout_oracle_summary.csv`, and `family_gate_summary.csv`: selected-candidate results.",
            "- `validation.json` and `manifest.json`: machine-readable checks and SHA-256 inventory.",
            "",
            f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, {platform.platform()}.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    required_rank = {
        (int(row["sample_id"]), row["method"])
        for row in rank
    }
    q24_success = family_success_by_q[("ordinary", 24)]
    checks = {
        "all_archived_iterations_match": bool(reproduction) and all(truth(row["exact_iteration_match"]) for row in reproduction),
        "all_eight_hard_ids_present": {int(row["sample_id"]) for row in rank} == set(ids),
        "all_rank_cases_present": all((sample_id, f"ordinary_m128_q{q}") in required_rank for sample_id in ids for q in q_requested),
        "strong_ilu_unchanged": all(reproduction_median(sample_id, "strong_ilu") == int(float(hardness[sample_id]["ilu_iterations_median"])) for sample_id in ids),
        "q24_majority_gate_passes": q24_success == 6,
        "q24_construction_robust_successes": sum(
            all(
                all(truth(value["converged"]) for value in group)
                and median(float(value["gamma"]) for value in group) >= settings.strong_gamma
                for group in (
                    [row for row in stability if int(row["sample_id"]) == sample_id and row["method"] == "ordinary_m128_q24_construction_stability" and int(row["construction_rhs_seed"]) == seed]
                    for seed in (settings.construction_rhs_seed,) + settings.stability_rhs_seeds
                )
            )
            for sample_id in ids
        ) == 6,
        "selection_and_heldout_disjoint": set(settings.selection_rhs_seeds).isdisjoint(settings.heldout_rhs_seeds),
        "local_global_correctly_not_triggered": (output / "local_global_diagnostics.csv").read_text(encoding="utf-8").strip() == "status\nno_rows",
        "invalid_noop_deletions_removed": not any(row.get("ablation") == "delete_from_q8" and int(row["q"]) == 8 for row in blocks),
        "no_neural_artifacts": not any(path.suffix in {".pt", ".pth"} for path in output.rglob("*")),
        "supporting_tests_recorded": (output / "test_results.txt").is_file(),
        "report_exists": report.is_file(),
        "required_plots_exist": all(
            (output / "plots" / name).is_file()
            for name in (
                "old_q8_residual_histories.png",
                "id15_mechanism_diagnostics.png",
                "rank_sensitivity_m128.png",
                "construction_stability_vs_solver.png",
            )
        ),
    }
    _write_json(
        output / "validation.json",
        {
            "assessment": "ready_to_share" if all(checks.values()) else "needs_revision",
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "oracle_gate": {
                "criterion": "all three held-out RHSs converge and median iteration reduction >= 0.40",
                "required_systems": settings.strict_majority,
                "ordinary_m128_requested_q24_successes": q24_success,
            },
            "caveat": "The recovered space is an expensive own-system oracle; no neural model was trained.",
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"final validation failed: {[key for key, value in checks.items() if not value]}")
    _write_manifest(output)

    # Add a compact post-classification provenance block without rewriting the
    # historical 758-file classification ledger.
    manifest = fd_root / "manifest.md"
    original = manifest.read_text(encoding="utf-8")
    marker = "\n## Post-classification research additions\n"
    if marker in original:
        original = original.split(marker, 1)[0].rstrip() + "\n"
    additions = [report] + sorted(path for path in output.rglob("*") if path.is_file())
    extra = [
        "",
        "## Post-classification research additions",
        "",
        "These RBF-FD artifacts were generated after the historical reorganization and are stored directly in the classified FD archive.",
        "",
        "| Filename | Location | Branch | Source experiment | Bytes | SHA-256 |",
        "|---|---|---|---|---:|---|",
    ]
    for path in additions:
        relative = path.relative_to(root)
        extra.append(
            f"| {path.name} | {relative} | RBF-FD | Stage 20 spatial-$K(x)$ robust oracle diagnostic | {path.stat().st_size} | {sha256(path)} |"
        )
    manifest.write_text(original.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
