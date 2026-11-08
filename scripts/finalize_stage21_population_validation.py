#!/usr/bin/env python3
"""Recompute Stage-21 summaries/figures from saved CSVs and write the report."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import scipy

from lcp_rbffd.experiments.stage21_population_validation import (
    Stage21Settings,
    _aggregate,
    _bool,
    _plot_results,
    _validate,
    _write_chart_map,
    write_manifest,
)
from lcp_rbffd.experiments.stage9_scalability import _write_json


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(value: Any) -> float:
    return float(value)


def i(value: Any) -> int:
    return int(float(value))


def fmt_break_even(row: Mapping[str, Any]) -> str:
    if not _bool(row["break_even_is_finite"]):
        return "--"
    value = str(i(row["break_even_rhs_count"]))
    if _bool(row["break_even_censored_baseline"]):
        return value + "$^{c}$"
    return value


def hard_table(rows: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "| ID | evidence cohort | ILU median | recycle median [min,max] | $J_{iter}$ | recycle convergence | actual $q$ | setup (s) | break-even RHS | gate |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in sorted(rows, key=lambda item: i(item["sample_id"])):
        cohort = "Stage 20" if row["provenance_cohort"] == "stage20_diagnostic" else "unseen"
        gate = "pass" if _bool(row["system_gate_passed"]) else "fail"
        lines.append(
            "| {id} | {cohort} | {base:.0f} | {rec:.0f} [{lo},{hi}] | {j:.3f} | {conv:.0%} | {q} | {setup:.3f} | {be} | {gate} |".format(
                id=i(row["sample_id"]),
                cohort=cohort,
                base=f(row["baseline_iterations_median"]),
                rec=f(row["recycle_iterations_median"]),
                lo=i(row["recycle_iterations_min"]),
                hi=i(row["recycle_iterations_max"]),
                j=f(row["J_iter_median"]),
                conv=f(row["recycle_convergence_rate"]),
                q=i(row["actual_q"]),
                setup=f(row["total_recycle_setup_seconds"]),
                be=fmt_break_even(row),
                gate=gate,
            )
        )
    return "\n".join(lines)


def easy_table(rows: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "| ID | ILU median | recycle median | absolute reduction | $J_{iter}$ | recycle time (s) | setup (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: i(item["sample_id"])):
        lines.append(
            "| {id} | {base:.0f} | {rec:.0f} | {absolute:.0f} | {j:.3f} | {solve:.3f} | {setup:.3f} |".format(
                id=i(row["sample_id"]),
                base=f(row["baseline_iterations_median"]),
                rec=f(row["recycle_iterations_median"]),
                absolute=f(row["iteration_absolute_reduction_median"]),
                j=f(row["J_iter_median"]),
                solve=f(row["recycle_solve_seconds_median"]),
                setup=f(row["total_recycle_setup_seconds"]),
            )
        )
    return "\n".join(lines)


def correlation_table(rows: Iterable[Mapping[str, Any]]) -> str:
    labels = {
        "d_Z": "$d_Z$",
        "d_W": "$d_W$",
        "max_principal_angle_Z_degrees": "max angle($Z$)",
        "max_principal_angle_W_degrees": "max angle($W$)",
    }
    hard = {row["metric"]: row for row in rows if row["scope"] == "hard"}
    rankings = {
        row["metric"]: row
        for row in rows
        if row["scope"] == "system_by_system_ranking"
    }
    lines = [
        r"| geometry metric | hard-pair Spearman $\rho$ | hard pairs | systems with defined within-system $\rho$ | median within-system $\rho$ | closest-pair agreement |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for metric in labels:
        global_row = hard[metric]
        ranking = rankings[metric]
        lines.append(
            "| {label} | {rho:.3f} | {pairs} | {systems} | {median:.3f} | {agreement:.1%} |".format(
                label=labels[metric],
                rho=f(global_row["spearman_coefficient"]),
                pairs=i(global_row["pair_count"]),
                systems=i(ranking["systems_with_defined_spearman"]),
                median=f(ranking["median_system_spearman"]),
                agreement=f(ranking["closest_pair_ranking_accuracy"]),
            )
        )
    return "\n".join(lines)


def write_report(
    report: Path,
    output: Path,
    settings: Stage21Settings,
    summary_rows: List[Mapping[str, Any]],
    correlation_rows: List[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> str:
    gate = aggregate["unseen_population_gate"]
    all_hard = aggregate["all_hard"]
    easy_summary = aggregate["easy_systems"]
    rank = aggregate["rank"]
    geometry_hard = aggregate["construction_geometry"]["hard"]
    geometry_unseen = aggregate["construction_geometry"]["unseen_hard"]
    functional = aggregate["construction_functional_stability"]
    timing = aggregate["timing"]
    hard = [row for row in summary_rows if row["difficulty_class"] == "hard"]
    easy = [row for row in summary_rows if row["difficulty_class"] == "easy"]
    unseen_hard = [row for row in hard if row["provenance_cohort"] == "unseen"]
    unseen_pass = [i(row["sample_id"]) for row in unseen_hard if _bool(row["system_gate_passed"])]
    unseen_fail = [i(row["sample_id"]) for row in unseen_hard if not _bool(row["system_gate_passed"])]
    corr_hard = {
        row["metric"]: row
        for row in correlation_rows
        if row["scope"] == "hard"
    }

    decision = (
        "A. Oracle validation passed; proceed to Stage 22 transfer."
        if gate["population_gate_passed"]
        else "B. Oracle validation failed; return to target construction."
    )

    lines = [
        "# Stage 21: population validation of the spatial-$K$ Ritz oracle",
        "",
        "## Technical summary and decision",
        "",
        f"**The frozen $m=128$, requested-$q=24$ ordinary-Ritz oracle does not pass the predeclared out-of-sample population gate.** It passes {gate['systems_passing']} of {gate['systems']} genuinely unseen hard systems ({gate['pass_fraction']:.1%}), below the required {settings.unseen_pass_fraction_gate:.0%}. The unseen-hard median recycled/baseline iteration ratio is {gate['median_iteration_ratio']:.3f}, which satisfies the separate $0.65$ ratio threshold, but both conditions were required. The successful unseen IDs are {', '.join(map(str, unseen_pass))}; the failures are {', '.join(map(str, unseen_fail))}. Across all 15 hard systems, including the eight that informed Stage 20, {all_hard['systems_passing']}/15 pass.",
        "",
        f"The useful spaces remain moderate rank: pair preservation gives actual rank {rank['actual_q_min']} or {rank['actual_q_max']}, only {100*rank['q_over_n_min']:.3f}%--{100*rank['q_over_n_max']:.3f}% of $N=5041$. Rank alone is therefore not the objection. The objection is robustness. The primary construction misses the frozen system gate on five hard systems, and the residual/test space $W=\operatorname{{orth}}(AZ)$ is not more construction-RHS stable than $Z$ in this population. Consequently, Stage 22 transfer is not justified.",
        "",
        f"**Decision: {decision}**",
        "",
        "## Frozen method, population, and gate",
        "",
        "All numerical choices were written to `predeclared_protocol.json` before the Stage-21 oracle run. The population is the complete saved Stage-19A sample: 36 numerically valid systems, of which 21 are easy and 15 are hard under archived strong-ILU median iteration counts. No system was invalid and none was excluded. The eight Stage-20 diagnostic IDs are 3, 7, 10, 14, 15, 17, 19, and 34. The other seven hard IDs, 0, 6, 8, 13, 16, 18, and 28, form the genuinely unseen decision cohort. No sampled system lies in the predeclared transition interval of 50--199 iterations.",
        "",
        "Every matrix uses $N=5041$ fixed seed-7 jittered nodes with jitter 0.15, stencil size 25, PHS $r^3$, degree-2 polynomial augmentation, and the archived smooth spatially varying anisotropic tensor $K(x)$. SuperLU ILU is unchanged with drop tolerance $10^{-3}$, fill factor 5, COLAMD ordering, and zero diagonal-pivot threshold. Right-preconditioned GMRES and recycled GMRES use restart 50, relative tolerance $10^{-8}$, and a 300-iteration cap. Stopping is based on the true residual.",
        "",
        "For construction RHS seed 20261001, Arnoldi is applied for 128 steps to $B=AM^{-1}$. Ordinary Ritz coordinates are sorted by increasing Ritz-value magnitude. Complete real directions and complex-conjugate blocks are retained without splitting. Each residual-coordinate vector $y_i$ is mapped to $z_i=M^{-1}y_i$, and $Z$ is orthonormalized. The paired nonsymmetric construction uses $AZ=WR$ and triangular solves for $U=ZR^{-1}$, hence $AU=W$ and $W^T W=I$. The fixed-space recycled iteration applies Arnoldi to $(I-WW^T)AM^{-1}$. Seeds 20261101--20261103 are held-out evaluation RHSs. Seeds 20261011 and 20261021 provide independent constructions only for the stability diagnostic.",
        "",
        "For each held-out RHS, the reported relative iteration reduction is $J_{iter}=(k_{ILU}-k_{recycle})/k_{ILU}$. A system passes only if all three recycled solves reach the true-residual tolerance and the median $J_{iter}$ is at least 0.40. The unseen population passes only if at least 70% of hard/transition systems pass and their median $k_{recycle}/k_{ILU}$ is at most 0.65. Capped iteration counts are retained at 300 under the established project convention and are not extrapolated.",
        "",
        "All 108 baseline solves reproduce the archived Stage-19A iteration and convergence results exactly. The 24 Stage-20 $m=128$, requested-$q=24$ held-out results also reproduce exactly. The maximum true relative residual among the 100 converged primary recycled solves is $%.3g$, below the $10^{-8}$ stopping threshold." % aggregate["true_residual_check"]["max_true_residual_among_converged"],
        "",
        "## The oracle helps many hard systems, but not enough unseen systems",
        "",
        "The scatter plot separates the eight systems that informed Stage 20 from the genuinely unseen systems. Points on the 300-iteration boundaries include capped failures. The frozen space substantially accelerates four unseen systems, but ID 0 reaches only 36.0% median reduction, ID 8 has only one converged primary held-out solve, and ID 16 fails all three primary solves.",
        "",
        "![Baseline versus recycle iterations](../stages/stage21_spatial_K_population_validation/plots/baseline_vs_recycle_iterations.png)",
        "",
        hard_table(hard),
        "",
        "$^{c}$ Break-even estimates use the fixed 300-step baseline budget because the strong-ILU baseline did not converge; they are censored timing comparisons, not converged time-to-solution estimates.",
        "",
        "The all-hard median iteration ratio is %.3f and median $J_{iter}$ is %.3f. This strong central tendency does not override the frozen population requirement: the unseen pass fraction is the failed condition. The distribution plot also shows why easy systems are kept separate." % (all_hard["median_iteration_ratio"], all_hard["median_J_iter"]),
        "",
        "![Oracle reduction distribution](../stages/stage21_spatial_K_population_validation/plots/oracle_reduction_distribution.png)",
        "",
        "## Easy systems do not establish useful oracle economics",
        "",
        f"The 21 easy systems have median strong-ILU count {easy_summary['median_baseline_iterations']:.0f} and median recycled count {easy_summary['median_recycle_iterations']:.0f}, an absolute median saving of only {easy_summary['median_absolute_reduction']:.0f} iterations. Only {easy_summary['systems_passing_40pct_gate']}/21 meet the relative 40% system threshold. More importantly, the roughly {timing['total_setup_seconds_median']:.3f}-second median recycle setup dwarfs an easy baseline solve, and projected-GMRES overhead makes the measured recycled online time larger than baseline time for every easy system. These cases support continued use of strong ILU alone.",
        "",
        easy_table(easy),
        "",
        "## Independent constructions do not reveal a more stable $W$ representation",
        "",
        f"Each system was reconstructed independently from three construction RHSs. All three constructions pass the system gate on {functional['hard_systems_passing_all_three_constructions']}/15 hard systems and {functional['unseen_hard_systems_passing_all_three_constructions']}/7 unseen hard systems. Thus the primary pass/fail pattern is not merely a single-construction accident, although ID 0 crosses the 40% threshold only for one alternate construction and ID 8 changes from primary cap hits to convergence under alternate constructions without reaching the gate.",
        "",
        "For bases of equal rank, the geometry diagnostics are $d_Z=\|Z_1Z_1^T-Z_2Z_2^T\|_F/\sqrt{2q}$ and $d_W=\|W_1W_1^T-W_2W_2^T\|_F/\sqrt{2q}$. The generalized normalization $\sqrt{q_1+q_2}$ is used for the occasional 23-versus-24 comparison. Among the 45 hard-system construction pairs, median $d_Z$ is %.3f and median $d_W$ is %.3f. $W$ is closer than $Z$ in only %.1f%% of pairs. Median maximum principal angle is %.1f degrees in $Z$ and %.1f degrees in $W$. On the unseen hard cohort alone, median $d_Z$ is %.3f and median $d_W$ is %.3f. The proposed explanation that large $Z$ variability hides a substantially more stable $W$ space is therefore contradicted." % (
            geometry_hard["d_Z_median"],
            geometry_hard["d_W_median"],
            100 * geometry_hard["d_W_less_than_d_Z_fraction"],
            geometry_hard["max_angle_Z_median_degrees"],
            geometry_hard["max_angle_W_median_degrees"],
            geometry_unseen["d_Z_median"],
            geometry_unseen["d_W_median"],
        ),
        "",
        "![Maximum principal angles in Z and W](../stages/stage21_spatial_K_population_validation/plots/max_principal_angle_Z_vs_W.png)",
        "",
        "Geometry is descriptively associated with construction-to-construction performance differences, but $W$ does not improve that association. The hard-pair Spearman coefficients are %.3f for $d_Z$ and %.3f for $d_W$. Maximum-angle correlations are %.3f for $Z$ and %.3f for $W$. Pair observations share systems and are not independent, so the coefficients are ranking diagnostics rather than inferential evidence. Only eight systems have nonconstant three-pair performance differences, making within-system correlations especially coarse." % (
            f(corr_hard["d_Z"]["spearman_coefficient"]),
            f(corr_hard["d_W"]["spearman_coefficient"]),
            f(corr_hard["max_principal_angle_Z_degrees"]["spearman_coefficient"]),
            f(corr_hard["max_principal_angle_W_degrees"]["spearman_coefficient"]),
        ),
        "",
        correlation_table(correlation_rows),
        "",
        "![Z distance versus performance difference](../stages/stage21_spatial_K_population_validation/plots/dZ_vs_performance_difference.png)",
        "",
        "![W distance versus performance difference](../stages/stage21_spatial_K_population_validation/plots/dW_vs_performance_difference.png)",
        "",
        "A representative disagreement remains: for successful ID 7, median pairwise $d_Z=0.362$ and $d_W=0.407$, with maximum angles near $81.2^\circ$ and $78.7^\circ$, yet the median iteration-ratio difference across constructions is only 0.040 and every construction passes. Conversely, ID 16 has similarly large distances, $d_Z=0.410$ and $d_W=0.480$, but every construction fails. Geometry alone therefore neither defines functional equivalence nor separates useful from useless recycle actions.",
        "",
        "## Setup cost can amortize only where the solver action works",
        "",
        f"Median Arnoldi/Ritz setup is {timing['ritz_setup_seconds_median']:.3f} s and median $AZ=WR$ pair setup is {timing['pair_setup_seconds_median']:.3f} s. Fifteen systems show a positive measured online-time saving, with median nominal break-even {timing['finite_break_even_median_rhs']:.0f} RHSs. These are the hard systems, and fourteen have capped ILU baselines, so their break-even values compare against a fixed failed 300-step budget and are optimistic. ID 15 is the only hard system with converged baseline and recycled solves; its measured incremental break-even is three RHSs. Failed recycled systems cannot claim an economic break-even even when their capped execution happens to be faster.",
        "",
        "![Setup and break-even summary](../stages/stage21_spatial_K_population_validation/plots/setup_cost_break_even.png)",
        "",
        "## Answers to the validation questions",
        "",
        "1. **Does $m=128$, $q=24$ generalize beyond the eight Stage-20 systems?** Not robustly under the predeclared standard. It passes 4/7 unseen hard systems, not the required 70%.",
        "2. **Is the useful oracle still low/moderate rank relative to $N$?** Yes where it works. Actual rank 23--24 is at most 0.476% of $N$, but this rank is not uniformly solver-useful.",
        "3. **Is $W$ more construction-RHS stable than $Z$?** No. Hard-pair median distance and maximum angle are slightly larger in $W$.",
        "4. **Does $W$ geometry track functional equivalence better than $Z$ geometry?** No. Hard-pair Spearman correlations and within-system ranking diagnostics are similar, with no consistent $W$ advantage.",
        "5. **Is the result strong enough for cross-system transfer?** No. The own-system frozen oracle fails the unseen population gate, so transfer would test a target that is not yet reliable on its own systems.",
        "",
        "## Limitations",
        "",
        "The population is the complete 36-system Stage-19A Latin-hypercube sample, not a new independent coefficient-field draw. The seven unseen hard systems were not used to choose $m=128$ or requested $q=24$, but their matrices share the same spatial-field distribution and discretization. There are only three construction RHSs and three held-out evaluation RHSs per system. The 300-iteration cap censors the hard baseline and some recycle failures. Timing was measured on one workstation and is more variable than iteration counts. Pairwise geometry observations within a system are dependent, so Spearman values are descriptive. No claim is made about other ranks, extraction rules, or discretizations because those choices were intentionally frozen.",
        "",
        "## Final decision",
        "",
        f"**{decision}**",
        "",
        "The single supported next step is to return to the complementary-target construction and require a new own-system oracle to pass this same full-population gate before attempting transfer or learning.",
        "",
        "## Reproducibility and saved evidence",
        "",
        "- `population_solver_results.csv`: all 324 construction/system/held-out-RHS combinations, including baseline and recycled metrics.",
        "- `system_summary.csv`: primary-construction system medians and gate outcomes.",
        "- `construction_results.csv`: Ritz and paired-space setup, rank, identities, and reduced conditioning.",
        "- `construction_geometry_diagnostics.csv`: all 108 construction-space pairs in $Z$ and $W$.",
        "- `construction_stability_summary.csv`: gate stability across all three constructions.",
        "- `geometry_performance_correlations.csv` and `system_geometry_ranking.csv`: aggregate and within-system ranking diagnostics.",
        "- `predeclared_protocol.json`, `validation.json`, and `manifest.json`: frozen settings, integrity checks, and hashes.",
        "- `test_results.txt`: 43 passing protocol, RBF-FD, nonsymmetric-pairing, Ritz, and recycled-GMRES tests.",
        "",
        f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, {platform.platform()}.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def write_source_notes(output: Path) -> None:
    text = """# Stage 21 source and QA notes

## Decision frame

- Question: does the frozen Stage-20 ordinary-Ritz oracle generalize to the full saved spatial-K population?
- Technical audience; system and unseen-population gates are defined in `predeclared_protocol.json`.
- Primary comparison: strong-ILU GMRES versus fixed-space recycled GMRES using the same matrix, ILU, RHS, stopping rule, and cap.

## Source inventory

- Stage-19A `coefficient_samples.csv`, `fixed_nodes.npz`, `hardness_scan_summary.csv`, and `hardness_scan_rhs.csv` define the complete saved population and archived baseline.
- Stage-20 `arnoldi_rank_ablation.csv` supplies the exact eight-system reproduction reference.
- `src/lcp_rbffd/experiments/stage21_population_validation.py` reconstructs matrices deterministically and runs the frozen oracle.
- `population_solver_results.csv` and `construction_geometry_diagnostics.csv` are the primary report evidence.

## Metric notes

- Capped counts remain 300 under the existing project convention.
- The primary population result uses construction seed 20261001 only. Seeds 20261011 and 20261021 are stability diagnostics.
- Break-even excludes ILU setup because ILU is common to both methods. Capped-baseline values are explicitly labeled censored.
- Pairwise Spearman correlations are descriptive because three pairs from the same system are dependent.

## Visual QA

- Six requested plots are rendered as static PNGs from saved CSVs.
- Equality and gate reference lines are shown where they define the comparison.
- Stage-20 and unseen systems use the same blue/orange mapping throughout.
- Dense per-system values remain in tables/CSVs rather than chart labels.
"""
    (output / "source_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = (
        root
        / "results"
        / "results-fd"
        / "stages"
        / "stage21_spatial_K_population_validation"
    )
    report = (
        root
        / "results"
        / "results-fd"
        / "reports"
        / "stage21_spatial_K_population_validation.md"
    )
    settings = Stage21Settings()
    per_rhs = read_rows(output / "population_solver_results.csv")
    constructions = read_rows(output / "construction_results.csv")
    geometry = read_rows(output / "construction_geometry_diagnostics.csv")
    summaries = read_rows(output / "system_summary.csv")

    aggregate = _aggregate(output, settings, per_rhs, geometry, summaries)
    _plot_results(output, summaries, geometry)
    _write_chart_map(output)
    correlation_rows = read_rows(output / "geometry_performance_correlations.csv")
    decision = write_report(
        report, output, settings, summaries, correlation_rows, aggregate
    )
    _write_json(
        output / "decision.json",
        {
            "decision": decision,
            "population_gate_passed": aggregate["unseen_population_gate"][
                "population_gate_passed"
            ],
            "proceed_to_stage22_transfer": False,
            "neural_training_performed": False,
        },
    )
    write_source_notes(output)
    _validate(
        root,
        output,
        report,
        settings,
        per_rhs,
        constructions,
        geometry,
        summaries,
    )
    write_manifest(output)
    print(report)


if __name__ == "__main__":
    main()
