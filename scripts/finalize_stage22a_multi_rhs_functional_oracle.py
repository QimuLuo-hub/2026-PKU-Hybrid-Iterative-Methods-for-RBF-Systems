#!/usr/bin/env python3
"""Validate saved Stage-22A evidence, render figures, and write the report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats

from lcp_rbffd.experiments.stage22a_multi_rhs_functional_oracle import (
    BLUE,
    CHARCOAL,
    GRAY,
    LIGHT_GRAY,
    OLIVE,
    ORANGE,
    PINK,
    PURPLE,
    Stage22ASettings,
    _bool,
)
from lcp_rbffd.experiments.stage9_scalability import _write_csv, _write_json


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def f(value: Any) -> float:
    return float(value)


def i(value: Any) -> int:
    return int(float(value))


def rule_map(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, int], Mapping[str, Any]]:
    return {(str(row["rule_id"]), i(row["sample_id"])): row for row in rows}


def system_ranking_diagnostics(
    geometry: Sequence[Mapping[str, Any]], output: Path
) -> List[Dict[str, Any]]:
    metrics = (
        "d_Z",
        "d_W",
        "max_principal_angle_Z_degrees",
        "max_principal_angle_W_degrees",
        "functional_curve_distance",
    )
    target_name = "median_iteration_ratio_difference"
    detailed: List[Dict[str, Any]] = []
    aggregate: List[Dict[str, Any]] = []
    sample_ids = sorted({i(row["sample_id"]) for row in geometry})
    for metric in metrics:
        defined = []
        closest = []
        for sample_id in sample_ids:
            local = [row for row in geometry if i(row["sample_id"]) == sample_id]
            x = np.asarray([f(row[metric]) for row in local])
            y = np.asarray([f(row[target_name]) for row in local])
            if len(local) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                result = stats.spearmanr(x, y)
                coefficient = float(result.statistic)
                pvalue = float(result.pvalue)
                defined.append(coefficient)
            else:
                coefficient = pvalue = float("nan")
            closest_ok = bool(y[int(np.argmin(x))] <= np.min(y) + 1.0e-12)
            closest.append(closest_ok)
            detailed.append(
                {
                    "sample_id": sample_id,
                    "metric": metric,
                    "spearman_coefficient": coefficient,
                    "spearman_pvalue": pvalue,
                    "pair_count": len(local),
                    "closest_pair_has_smallest_performance_difference": closest_ok,
                }
            )
        aggregate.append(
            {
                "metric": metric,
                "systems": len(sample_ids),
                "systems_with_defined_spearman": len(defined),
                "median_system_spearman": float(np.median(defined)) if defined else float("nan"),
                "closest_pair_ranking_accuracy": float(np.mean(closest)),
            }
        )
    _write_csv(output / "geometry_functional_system_rankings.csv", detailed)
    _write_csv(output / "geometry_functional_ranking_summary.csv", aggregate)
    return aggregate


def diagnose_systems(
    summaries: Sequence[Mapping[str, Any]],
    singular: Sequence[Mapping[str, Any]],
    output: Path,
) -> List[Dict[str, Any]]:
    lookup = rule_map(summaries)
    rows: List[Dict[str, Any]] = []
    ids = sorted({i(row["sample_id"]) for row in summaries})
    candidate_rules = sorted(
        {
            str(row["rule_id"])
            for row in summaries
            if str(row["method"]) in {"Z_union", "W_union", "functional_greedy"}
        }
    )
    for sample_id in ids:
        stage21 = lookup[("Z_union_n1_m128_q24", sample_id)]
        deep = lookup[("cost_1rhs_m256_q24", sample_id)]
        robust = lookup[("Z_union_n3_m128_q48", sample_id)]
        passing = [
            row
            for row in summaries
            if i(row["sample_id"]) == sample_id
            and str(row["rule_id"]) in candidate_rules
            and _bool(row["system_gate_passed"])
        ]
        passing.sort(
            key=lambda row: (
                i(row["actual_q"]),
                f(row["deployable_setup_seconds"]),
                -f(row["J_iter_median"]),
            )
        )
        smallest = passing[0] if passing else None
        ranks = {}
        for n_construct in (1, 2, 3, 4):
            local = [
                row
                for row in singular
                if i(row["sample_id"]) == sample_id
                and row["space"] == "Z_raw"
                and i(row["n_construct"]) == n_construct
            ]
            ranks[n_construct] = i(local[0]["numerical_rank"])
        if _bool(stage21["system_gate_passed"]):
            diagnosis = "Stage-21 single-RHS oracle already passed"
        elif _bool(deep["system_gate_passed"]):
            diagnosis = "single-RHS Arnoldi depth/quality, not RHS diversity"
        elif _bool(robust["system_gate_passed"]):
            diagnosis = "moderate-rank union required"
        else:
            diagnosis = "ordinary-Ritz target not repaired"
        rows.append(
            {
                "sample_id": sample_id,
                "stage21_passed": _bool(stage21["system_gate_passed"]),
                "single_rhs_m256_q24_passed": _bool(deep["system_gate_passed"]),
                "three_rhs_m128_q48_passed": _bool(robust["system_gate_passed"]),
                "smallest_passing_rule_diagnostic": smallest["rule_id"] if smallest else "",
                "smallest_passing_actual_q": i(smallest["actual_q"]) if smallest else "",
                "Z_dictionary_rank_n1": ranks[1],
                "Z_dictionary_rank_n2": ranks[2],
                "Z_dictionary_rank_n3": ranks[3],
                "Z_dictionary_rank_n4": ranks[4],
                "new_rank_n1_to_n4": ranks[4] - ranks[1],
                "diagnosis": diagnosis,
            }
        )
    _write_csv(output / "system_failure_diagnosis.csv", rows)
    return rows


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_results(
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    family: Sequence[Mapping[str, Any]],
) -> None:
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    lookup = rule_map(summaries)
    ids = sorted({i(row["sample_id"]) for row in summaries})
    preferred = [lookup[("Z_union_n3_m128_q48", sample_id)] for sample_id in ids]

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    baseline = np.asarray([f(row["baseline_iterations_median"]) for row in preferred])
    recycle = np.asarray([f(row["recycle_iterations_median"]) for row in preferred])
    ax.scatter(baseline, recycle, color=BLUE, edgecolor="white", linewidth=0.7, s=58)
    for sample_id, x, y in zip(ids, baseline, recycle):
        if sample_id in {0, 8, 10, 15, 16, 17}:
            ax.annotate(str(sample_id), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.plot([0, 305], [0, 305], color=GRAY, linestyle="--", linewidth=1.2)
    ax.plot([0, 305], [0, 183], color=OLIVE, linestyle=":", linewidth=1.2)
    ax.set(xlim=(0, 310), ylim=(0, 310), xlabel="Strong-ILU GMRES iterations (median)", ylabel="Three-RHS Z-union recycled iterations (median)")
    ax.set_title("Baseline and repaired-oracle iterations", pad=30)
    ax.text(0.01, 1.01, "15 Stage-21 hard systems; requested q=48; three held-out RHS per system", transform=ax.transAxes, fontsize=9, color=GRAY)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    _save(fig, plots / "baseline_vs_recycle_iterations.png")

    reductions = np.asarray([f(row["J_iter_median"]) for row in preferred])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(reductions, bins=np.linspace(0.4, 0.95, 12), color=BLUE, edgecolor="white")
    ax.axvline(0.40, color=ORANGE, linestyle="--", linewidth=1.4, label="system gate")
    ax.set(xlabel="Median relative iteration reduction $J_{iter}$", ylabel="Systems")
    ax.set_title("Repaired-oracle reduction distribution", pad=30)
    ax.text(0.01, 1.01, "Three-RHS Z-union, requested q=48; all 15 systems pass", transform=ax.transAxes, fontsize=9, color=GRAY)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    _save(fig, plots / "oracle_reduction_distribution.png")

    for metric, label, name, color in (
        ("d_Z", "$d_Z$", "dZ_vs_performance_difference.png", BLUE),
        ("d_W", "$d_W$", "dW_vs_performance_difference.png", ORANGE),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.scatter(
            [f(row[metric]) for row in geometry],
            [f(row["median_iteration_ratio_difference"]) for row in geometry],
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set(xlabel=label, ylabel="Absolute median iteration-ratio difference")
        ax.set_title(f"{label} and construction-performance difference", pad=30)
        ax.text(0.01, 1.01, "90 independent-construction pairs across 15 hard systems", transform=ax.transAxes, fontsize=9, color=GRAY)
        ax.grid(color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
        _save(fig, plots / name)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    x = np.asarray([f(row["max_principal_angle_Z_degrees"]) for row in geometry])
    y = np.asarray([f(row["max_principal_angle_W_degrees"]) for row in geometry])
    ax.scatter(x, y, color=PURPLE, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.plot([0, 92], [0, 92], color=GRAY, linestyle="--", linewidth=1.2)
    ax.set(xlim=(0, 92), ylim=(0, 92), xlabel="Maximum principal angle in Z (degrees)", ylabel="Maximum principal angle in W (degrees)")
    ax.set_title("Construction-RHS principal angles", pad=30)
    ax.text(0.01, 1.01, "Same-system q≈24 spaces; diagonal denotes equal Z/W angle", transform=ax.transAxes, fontsize=9, color=GRAY)
    ax.grid(color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    _save(fig, plots / "max_principal_angle_Z_vs_W.png")

    chosen_rules = (
        "cost_1rhs_m128_q24",
        "cost_1rhs_m256_q24",
        "Z_union_n3_m128_q48",
        "Z_union_n4_m128_q64",
        "functional_greedy_n4_m128_q64",
    )
    labels = ("1×128, q24", "1×256, q24", "3×128, q48", "4×128, q64", "greedy, q64")
    fam = {row["rule_id"]: row for row in family}
    setups = [f(fam[rule]["median_oracle_diagnostic_setup_seconds"]) for rule in chosen_rules]
    break_even = []
    for rule in chosen_rules:
        finite = [
            f(row["break_even_rhs_oracle_diagnostic"])
            for row in summaries
            if row["rule_id"] == rule
            and math.isfinite(f(row["break_even_rhs_oracle_diagnostic"]))
        ]
        break_even.append(float(np.median(finite)))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    positions = np.arange(len(labels))
    axes[0].bar(positions, setups, color=[GRAY, ORANGE, BLUE, OLIVE, PURPLE])
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, labels, rotation=28, ha="right")
    axes[0].set(ylabel="Median setup / selection time (s, log scale)", title="Construction cost")
    axes[1].bar(positions, break_even, color=[GRAY, ORANGE, BLUE, OLIVE, PURPLE])
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, labels, rotation=28, ha="right")
    axes[1].set(ylabel="Median break-even RHS (log scale)", title="Repeated-RHS break-even")
    for ax in axes:
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7, alpha=0.7)
    fig.suptitle("Setup cost and repeated-RHS break-even", y=1.02)
    _save(fig, plots / "setup_cost_break_even_summary.png")


def _table_system(rows: Sequence[Mapping[str, Any]], rule_id: str) -> str:
    selected = sorted((row for row in rows if row["rule_id"] == rule_id), key=lambda row: i(row["sample_id"]))
    lines = [
        "| ID | ILU median | recycle median [min,max] | actual q | $J_{iter}$ | convergence | setup (s) | online (s) | break-even RHS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        be = f(row["break_even_rhs_deployable"])
        be_text = "--" if not math.isfinite(be) else str(int(be)) + ("$^c$" if _bool(row["break_even_censored_baseline"]) else "")
        lines.append(
            "| {id} | {base:.0f} | {rec:.0f} [{lo},{hi}] | {q} | {j:.3f} | {conv:.0%} | {setup:.3f} | {online:.3f} | {be} |".format(
                id=i(row["sample_id"]), base=f(row["baseline_iterations_median"]),
                rec=f(row["recycle_iterations_median"]), lo=i(row["recycle_iterations_min"]),
                hi=i(row["recycle_iterations_max"]), q=i(row["actual_q"]),
                j=f(row["J_iter_median"]), conv=f(row["convergence_rate"]),
                setup=f(row["deployable_setup_seconds"]), online=f(row["online_solve_seconds_median"]), be=be_text,
            )
        )
    return "\n".join(lines)


def _table_rules(family: Sequence[Mapping[str, Any]], rule_ids: Sequence[str]) -> str:
    lookup = {row["rule_id"]: row for row in family}
    lines = [
        "| fixed rule | systems passing | median iteration ratio | median actual q | median setup (s) |",
        "|:---|---:|---:|---:|---:|",
    ]
    for rule_id in rule_ids:
        row = lookup[rule_id]
        lines.append(
            f"| `{rule_id}` | {i(row['systems_passing'])}/15 | {f(row['median_iteration_ratio']):.3f} | {f(row['median_actual_q']):.0f} | {f(row['median_oracle_diagnostic_setup_seconds']):.3f} |"
        )
    return "\n".join(lines)


def write_report(
    report: Path,
    output: Path,
    summaries: Sequence[Mapping[str, Any]],
    family: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
    diagnoses: Sequence[Mapping[str, Any]],
) -> None:
    fam = {row["rule_id"]: row for row in family}
    corr = {row["metric"]: row for row in correlations}
    ranking = {row["metric"]: row for row in rankings}
    robust = fam["Z_union_n3_m128_q48"]
    maximal = fam["Z_union_n4_m128_q64"]
    greedy = fam["functional_greedy_n4_m128_q64"]
    w_robust = fam["W_union_n3_m128_q48"]
    deep = fam["cost_1rhs_m256_q24"]
    z_ranks = [
        i(row["actual_q"])
        for row in summaries
        if row["rule_id"] == "Z_union_n3_m128_q48"
    ]
    setup_row = next(
        row for row in read_rows(output / "setup_results.csv")
        if row["rule_id"] == "Z_union_n3_m128_q48"
    )
    stage21_failures = [row for row in diagnoses if not _bool(row["stage21_passed"])]
    depth_rescues = [i(row["sample_id"]) for row in stage21_failures if _bool(row["single_rhs_m256_q24_passed"])]
    rank_rescues = [i(row["sample_id"]) for row in stage21_failures if not _bool(row["single_rhs_m256_q24_passed"]) and _bool(row["three_rhs_m128_q48_passed"])]
    geometry_z = np.asarray([f(row["d_Z"]) for row in geometry])
    geometry_w = np.asarray([f(row["d_W"]) for row in geometry])
    angle_z = np.asarray([f(row["max_principal_angle_Z_degrees"]) for row in geometry])
    angle_w = np.asarray([f(row["max_principal_angle_W_degrees"]) for row in geometry])

    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 22A: multi-RHS functional oracle for spatially varying $K(x)$",
        "",
        "## Technical summary and decision",
        "",
        "**A robust own-system ordinary-Ritz oracle is recovered, but not because same-rank multi-RHS diversity fixes the Stage-21 construction.** The simplest all-system repair is a three-construction-RHS $Z$-union with $m=128$ per RHS and requested rank 48. It passes 15/15 hard systems and all 45 held-out solves. Median recycled/baseline iteration ratio is %.3f and median $J_{iter}$ is %.3f. Actual rank is %d--%d, at most %.3f%% of $N=5041$. A four-RHS requested-$q=64$ union also passes 15/15 and lowers the median ratio to %.3f, but uses more setup and rank." % (
            f(robust["median_iteration_ratio"]), f(robust["median_J_iter"]), min(z_ranks), max(z_ranks), 100 * max(z_ranks) / 5041, f(maximal["median_iteration_ratio"])
        ),
        "",
        "The mechanism diagnosis is more specific. At fixed final rank 24, two $m=128$ construction RHSs pass only 10/15 systems and four $m=64$ RHSs pass only 1/15, whereas one deeper $m=256$ construction passes 14/15. The matched nominal Arnoldi work therefore favors depth over RHS diversity. The four Stage-21 failures %s are rescued by the deeper single-RHS construction; ID %s still needs the moderate-rank union. Functional greedy passes all systems but requires a median %.1f s of oracle selection work, versus %.3f s for the three-RHS $Z$-union, without better held-out iterations. The stable QR-repaired $W$-union matches the high-rank $Z$-union but does not beat it." % (
            ", ".join(map(str, depth_rescues)), ", ".join(map(str, rank_rescues)), f(greedy["median_oracle_diagnostic_setup_seconds"]), f(robust["median_oracle_diagnostic_setup_seconds"])
        ),
        "",
        "**Recommendation A. Robust own-system oracle recovered; proceed to cross-system transfer.**",
        "",
        "## Frozen problem, construction, and gates",
        "",
        "The experiment uses the 15 hard systems identified solely by archived Stage-19A strong-ILU medians. These are IDs 0, 3, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 28, and 34. No transition system exists in the saved population. Matrices use the fixed $N=5041$ seed-7 jittered node set, jitter 0.15, stencil size 25, PHS $r^3$, degree-2 polynomial augmentation, and the saved smooth spatially varying anisotropic $K(x)$. ILU remains SuperLU with drop tolerance $10^{-3}$, fill factor 5, COLAMD ordering, and zero diagonal-pivot threshold. Right-preconditioned GMRES and recycled GMRES use restart 50, true-relative-residual tolerance $10^{-8}$, and cap 300.",
        "",
        "Construction RHS seeds are 20261001, 20261011, 20261021, and 20261031. Selection seeds 20261041 and 20261051 are used only by functional greedy. Final evaluation seeds 20261101--20261103 are disjoint from both. For each construction RHS, 128 Arnoldi steps are applied to $B=AM^{-1}$. Pair-preserving ordinary Ritz directions are ordered by increasing Ritz-value magnitude, mapped by $z=M^{-1}V_mg$, and retained up to 24 directions. The raw blocks are concatenated without system-specific tuning.",
        "",
        r"The $Z$-union takes the leading left singular vectors of the concatenated solution-space dictionary. The $W$-union takes the leading left singular vectors of $AZ_{raw}$ and maps them back within $\operatorname{range}(Z_{raw})$. A second thin QR and triangular solve is required after this back-map to enforce $AU=W$ stably; after repair, the maximum measured relative $AU-W$ defect over all W-union spaces is below $1.5\times10^{-12}$. Functional greedy preserves the within-RHS pair order and chooses among the next exposed real/complex blocks using actual recycled-GMRES performance on the two selection RHSs. Every final space uses the same fixed nonsymmetric projected solver with $AU=W$ and Arnoldi on $(I-WW^T)AM^{-1}$.",
        "",
        "A system passes only when all three held-out RHSs reach true relative residual at most $10^{-8}$ and median $J_{iter}=(k_{ILU}-k_{recycle})/k_{ILU}$ is at least 0.40. One fixed rule passes the family gate only at 12/15 systems or more with requested rank at most 64. Requested rank and actual numerical rank are kept separate. Capped baselines remain at 300 under the project convention.",
        "",
        "## A moderate-rank union repairs every hard system",
        "",
        "The three-RHS requested-$q=48$ $Z$-union is the lowest-rank Z-union rule that passes every system. Its actual rank falls below 48 when multiple construction spaces are numerically redundant, most strongly on systems already easy for the oracle. All 45 held-out solves converge, with median system reductions from %.1f%% to %.1f%%. The five Stage-21 failures or marginals are no longer exceptional: IDs 0, 8, 10, 16, and 17 have median recycled counts 35, 35, 38, 46, and 91, respectively, compared with baseline caps of 300." % (
            100 * min(f(row["J_iter_median"]) for row in summaries if row["rule_id"] == "Z_union_n3_m128_q48"),
            100 * max(f(row["J_iter_median"]) for row in summaries if row["rule_id"] == "Z_union_n3_m128_q48"),
        ),
        "",
        "![Baseline and repaired-oracle iterations](../stages/stage22A_spatial_K_multi_rhs_functional_oracle/plots/baseline_vs_recycle_iterations.png)",
        "",
        _table_system(summaries, "Z_union_n3_m128_q48"),
        "",
        "$^c$ Break-even counts are censored because 14 of the 15 ILU baselines hit the 300-step cap; they compare setup against a fixed failed-solve budget, not converged time-to-solution. ID 15 is the only uncensored hard baseline.",
        "",
        "![Oracle reduction distribution](../stages/stage22A_spatial_K_multi_rhs_functional_oracle/plots/oracle_reduction_distribution.png)",
        "",
        "The fixed-rule comparison separates construction mechanism from rank. $Z$-union and the numerically repaired $W$-union both pass 15/15 at three RHSs and requested $q=48$, with the same median iteration ratio %.3f. A four-RHS requested-$q=64$ union is slightly faster online but raises median setup from %.3f s to %.3f s. Functional greedy also passes every system, but its selection cost is two orders of magnitude larger and its median iteration ratio %.3f is slightly worse than the full high-rank union." % (
            f(w_robust["median_iteration_ratio"]), f(robust["median_oracle_diagnostic_setup_seconds"]), f(maximal["median_oracle_diagnostic_setup_seconds"]), f(greedy["median_iteration_ratio"])
        ),
        "",
        _table_rules(family, (
            "Z_union_n1_m128_q24", "Z_union_n2_m128_q32", "Z_union_n3_m128_q48", "W_union_n3_m128_q48", "Z_union_n4_m128_q64", "W_union_n4_m128_q64", "functional_greedy_n4_m128_q64"
        )),
        "",
        "No harmonic secondary was run. This follows the predeclared stop: ordinary Ritz already yields a fixed rule with zero persistent failures, so a harmonic or ordinary-plus-harmonic sweep would not diagnose a remaining failure.",
        "",
        "## Matched work shows that Arnoldi depth matters more than RHS diversity",
        "",
        "At the common final rank 24, nominal Arnoldi work is 256 steps for one RHS at $m=256$, two RHSs at $m=128$, and four RHSs at $m=64$. The single deep run passes 14/15 systems with median iteration ratio %.3f. Two $m=128$ runs pass 10/15 with ratio %.3f. Four shallow $m=64$ runs pass only 1/15 and have median ratio 1.000. The Stage-21 one-$m=128$ baseline also passes 10/15. Thus additional RHSs do add numerical directions, but same-rank diversity is not the effective resource. Accurate spectral resolution from a sufficiently deep Arnoldi relation is decisive." % (
            f(deep["median_iteration_ratio"]), f(fam["cost_2rhs_m128_q24"]["median_iteration_ratio"])
        ),
        "",
        _table_rules(family, (
            "cost_1rhs_m128_q24", "cost_1rhs_m256_q24", "cost_2rhs_m128_q24", "cost_4rhs_m64_q24"
        )),
        "",
        "The raw $Z$ dictionary's numerical rank grows from a median 24 for one RHS to 41, 51, and 58 for two, three, and four RHSs. The corresponding ranges are 23--24, 31--47, 32--71, and 33--95. Hence independent RHSs often contribute genuinely new directions. That fact alone does not establish causal solver value, because truncating those unions back to rank 24 does not improve the family pass count. Retaining a moderate rank of about 48 does.",
        "",
        "The stubborn-system ablation reinforces the distinction. IDs %s are repaired by one deeper $m=256$, $q=24$ construction. ID %s remains at 228 iterations and only 24%% reduction under that rule, but falls to 91 iterations and 69.7%% reduction under the three-RHS requested-$q=48$ union. The evidence therefore attributes four failures to Arnoldi depth/quality and the last to required solver-critical dimension. It does not support failure of ordinary Ritz as a target on this own-system family." % (
            ", ".join(map(str, depth_rescues)), ", ".join(map(str, rank_rescues))
        ),
        "",
        "## Functional distance is the strongest of the tested equivalence diagnostics",
        "",
        "Four independent single-RHS $m=128$, requested-$q=24$ spaces give 90 within-system construction pairs. Geometry remains variable: median $d_Z=%.3f$, median $d_W=%.3f$, and median maximum angles are %.1f degrees in $Z$ and %.1f degrees in $W$. The normalized functional distance is the RMS difference between true-residual curves in clipped $\log_{10}$ scale on the common 0--300 iteration grid, divided by eight decades and averaged over the three held-out RHSs." % (
            float(np.median(geometry_z)), float(np.median(geometry_w)), float(np.median(angle_z)), float(np.median(angle_w))
        ),
        "",
        "Across all pairs, Spearman correlation with absolute median iteration-ratio difference is %.3f for functional distance, compared with %.3f for $d_Z$ and %.3f for $d_W$. Maximum-angle correlations are %.3f for $Z$ and %.3f for $W$. Functional distance is therefore the best ranking diagnostic in this experiment, but the improvement is moderate rather than a clean separation. Within systems, its median defined Spearman coefficient is %.3f and its closest-pair ranking accuracy is %.1f%%. Because all pair observations reuse systems and evaluation RHSs, these are descriptive diagnostics, not independent inferential samples." % (
            f(corr["functional_curve_distance"]["spearman_coefficient"]), f(corr["d_Z"]["spearman_coefficient"]), f(corr["d_W"]["spearman_coefficient"]), f(corr["max_principal_angle_Z_degrees"]["spearman_coefficient"]), f(corr["max_principal_angle_W_degrees"]["spearman_coefficient"]), f(ranking["functional_curve_distance"]["median_system_spearman"]), 100 * f(ranking["functional_curve_distance"]["closest_pair_ranking_accuracy"])
        ),
        "",
        "![Z geometry and functional performance difference](../stages/stage22A_spatial_K_multi_rhs_functional_oracle/plots/dZ_vs_performance_difference.png)",
        "",
        "![W geometry and functional performance difference](../stages/stage22A_spatial_K_multi_rhs_functional_oracle/plots/dW_vs_performance_difference.png)",
        "",
        "![Maximum principal angles in Z and W](../stages/stage22A_spatial_K_multi_rhs_functional_oracle/plots/max_principal_angle_Z_vs_W.png)",
        "",
        "The Stage-21 hypothesis that $W$ geometry is systematically more stable remains unsupported. The stabilized $W$ construction is algebraically valid and solver-effective at high rank, but independent-construction $d_W$ is not smaller than $d_Z$ and its global correlation is not stronger. Functional trajectory equivalence is more informative than either projector distance.",
        "",
        "## Cost and deployability",
        "",
        "For the recommended three-RHS requested-$q=48$ rule, median setup is %.3f s and median online solve time is %.3f s. The setup performs nominally 384 Arnoldi $A$ applications, %d construction applications of $M^{-1}$, and up to 48 additional $A$ applications for pairing; exact counts and actual ranks are retained per system. Median estimated break-even is six held-out RHSs, but 14 values are censored by failed 300-step ILU baselines. The four-RHS $q=64$ rule has median break-even eight. Functional greedy's median diagnostic setup is %.1f s, so it is an expensive oracle diagnostic rather than a deployable construction." % (
            f(robust["median_oracle_diagnostic_setup_seconds"]),
            float(np.median([f(row["online_solve_seconds_median"]) for row in summaries if row["rule_id"] == "Z_union_n3_m128_q48"])),
            i(setup_row["construction_m_applications"]),
            f(greedy["median_oracle_diagnostic_setup_seconds"]),
        ),
        "",
        "![Setup cost and family robustness](../stages/stage22A_spatial_K_multi_rhs_functional_oracle/plots/setup_cost_break_even_summary.png)",
        "",
        "This remains an own-system oracle: constructing three or four spaces requires matrix-specific Arnoldi work. The result establishes a solver-relevant target and an upper bound, not an economical predictor. The next stage must test reuse without changing this construction rule and must keep oracle setup separate from online cost.",
        "",
        "## Answers to the Stage 22A questions",
        "",
        "1. **Is single-RHS excitation the main cause of Stage-21 failure?** No. Same-rank multi-RHS unions do not improve the 10/15 pass count. One deeper single-RHS Arnoldi run passes 14/15, showing that spectral resolution is more important. ID 17 additionally requires higher rank.",
        "2. **Does multi-RHS construction recover a robust moderate-rank oracle?** Yes. The fixed three-RHS $m=128$ requested-$q=48$ Z-union passes 15/15 systems, with actual rank 32--48.",
        "3. **Is diversity better than depth at matched cost?** No. At 256 nominal Arnoldi steps and final rank 24, one $m=256$ RHS decisively outperforms two $m=128$ and four $m=64$ RHSs.",
        "4. **Which construction works best?** Z- and stably paired W-union are essentially tied at sufficient rank. Z-union is the cleanest fixed construction. Functional greedy adds no held-out benefit commensurate with its selection cost.",
        "5. **Can the stubborn systems be rescued?** Yes. IDs 0, 8, 10, and 16 pass with one $m=256$, $q=24$ construction. ID 17 passes with the three-RHS requested-$q=48$ union. All pass under the fixed union rule.",
        "6. **Does functional similarity explain solver equivalence better than geometry?** Modestly. Its global Spearman coefficient is 0.743 versus 0.679 for $d_Z$ and 0.676 for $d_W$, but it is not a perfect surrogate.",
        "7. **Is there one fixed family-level rule?** Yes. Three independent $m=128$ ordinary-Ritz constructions, Z-union compression, and requested $q=48$ is fixed across all systems and passes all 15.",
        "",
        "## Limitations",
        "",
        "The study reuses the 15 hard systems from the finite Stage-19A population rather than drawing a new coefficient-field sample. Four construction RHSs and three final RHSs limit the stability denominator. Fourteen ILU baselines are capped, so reductions and break-even values use the project's censored convention. The family-rule grid contains multiple requested ranks, and the recommended rule is selected from that predeclared grid. Functional greedy is selection-RHS optimized but evaluated on disjoint RHSs. Timings come from one workstation and are less stable than iteration counts. No cross-system transfer or neural training was performed.",
        "",
        "## Final recommendation",
        "",
        "**A. Robust own-system oracle recovered; proceed to cross-system transfer.**",
        "",
        "The single next step is to freeze the three-RHS $m=128$, requested-$q=48$ Z-union rule and test neighboring spatial-$K$ reuse. No learning experiment should precede that transfer test.",
        "",
        "## Reproducibility and saved evidence",
        "",
        "- `solver_results.csv`: all per-system, per-held-out-RHS solver outcomes for the main, greedy, stability, and matched-cost experiments.",
        "- `system_rule_summary.csv` and `family_rule_summary.csv`: system gates and fixed-rule family gates.",
        "- `construction_results.csv` and `setup_results.csv`: construction timing, operation counts, attainable rank, pairing identities, and conditioning.",
        "- `dictionary_singular_values.csv`: numerical-rank and redundancy evidence as construction RHSs are added.",
        "- `greedy_search_trials.csv`: every selection-RHS greedy decision and its actual solver score.",
        "- `residual_histories.csv`: true residual histories for every final evaluation solve.",
        "- `geometry_functional_diagnostics.csv`: independent-construction Z/W geometry and functional curve distance.",
        "- `predeclared_protocol.json`, `validation.json`, and `manifest.json`: frozen protocol, integrity checks, and artifact hashes.",
        "",
        f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, {platform.platform()}.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(
    output: Path,
    report: Path,
    settings: Stage22ASettings,
    solver: Sequence[Mapping[str, Any]],
    setup: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    family: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    full_rules = [row for row in family if i(row["systems_evaluated"]) == 15]
    checks["three_rhs_q48_has_45_rows"] = sum(row["rule_id"] == "Z_union_n3_m128_q48" for row in solver) == 45
    checks["recommended_all_converged"] = all(
        _bool(row["recycle_converged"])
        for row in solver if row["rule_id"] == "Z_union_n3_m128_q48"
    )
    checks["recommended_true_residual_below_tolerance"] = max(
        f(row["recycle_true_final_relative_residual"])
        for row in solver if row["rule_id"] == "Z_union_n3_m128_q48"
    ) <= settings.solver.relative_tolerance
    checks["recommended_passes_15_systems"] = sum(
        row["rule_id"] == "Z_union_n3_m128_q48" and _bool(row["system_gate_passed"])
        for row in summaries
    ) == 15
    checks["family_gate_denominator_complete"] = all(i(row["systems_evaluated"]) == 15 for row in full_rules)
    checks["all_pair_identities_stable"] = max(f(row["au_equals_w_relative_error"]) for row in setup) < 1.0e-10
    checks["all_w_orthogonality_stable"] = max(f(row["test_orthogonality_error"]) for row in setup) < 1.0e-10
    checks["rhs_splits_disjoint"] = not (
        set(settings.construction_rhs_seeds) & set(settings.selection_rhs_seeds)
        or set(settings.construction_rhs_seeds) & set(settings.heldout_rhs_seeds)
        or set(settings.selection_rhs_seeds) & set(settings.heldout_rhs_seeds)
    )
    checks["report_exists"] = report.exists() and report.stat().st_size > 0
    checks["six_required_figures_exist"] = all(
        (output / "plots" / name).exists()
        for name in (
            "baseline_vs_recycle_iterations.png",
            "oracle_reduction_distribution.png",
            "dZ_vs_performance_difference.png",
            "dW_vs_performance_difference.png",
            "max_principal_angle_Z_vs_W.png",
            "setup_cost_break_even_summary.png",
        )
    )
    checks["no_neural_artifacts"] = not any(
        path.suffix.lower() in {".pt", ".pth", ".onnx"}
        for path in output.rglob("*") if path.is_file()
    )
    secondary_rows = read_rows(output / "secondary_solver_results.csv")
    checks["no_secondary_sweep_when_ordinary_has_no_failures"] = (
        not secondary_rows
        or all(row.get("status") == "no_rows" for row in secondary_rows)
    )
    status = "passed" if all(checks.values()) else "failed"
    validation = {
        "status": status,
        "checks": checks,
        "maximum_pair_identity_error": max(f(row["au_equals_w_relative_error"]) for row in setup),
        "maximum_true_residual_among_converged": max(
            f(row["recycle_true_final_relative_residual"])
            for row in solver if _bool(row["recycle_converged"])
        ),
    }
    _write_json(output / "validation.json", validation)
    if status != "passed":
        raise RuntimeError(f"Stage-22A validation failed: {checks}")
    return validation


def write_supporting_files(output: Path, report: Path) -> None:
    chart_rows = [
        {"figure": "baseline_vs_recycle_iterations.png", "question": "Does the repaired fixed oracle reduce hard-system iterations?", "source": "system_rule_summary.csv", "form": "paired scatter"},
        {"figure": "oracle_reduction_distribution.png", "question": "How broadly does the fixed oracle clear the 40% gate?", "source": "system_rule_summary.csv", "form": "histogram"},
        {"figure": "dZ_vs_performance_difference.png", "question": "Does Z geometry track functional construction differences?", "source": "geometry_functional_diagnostics.csv", "form": "scatter"},
        {"figure": "dW_vs_performance_difference.png", "question": "Does W geometry track functional construction differences?", "source": "geometry_functional_diagnostics.csv", "form": "scatter"},
        {"figure": "max_principal_angle_Z_vs_W.png", "question": "Is W more construction-RHS stable than Z?", "source": "geometry_functional_diagnostics.csv", "form": "paired scatter"},
        {"figure": "setup_cost_break_even_summary.png", "question": "Which robust construction has acceptable oracle cost?", "source": "family_rule_summary.csv", "form": "two-panel bars"},
    ]
    _write_csv(output / "chart_map.csv", chart_rows)
    source_notes = """# Stage 22A source and QA notes

## Decision frame

- Technical question: can independent construction RHSs repair the failed Stage-21 own-system oracle without changing the RBF-FD matrix, ILU, or recycled solver?
- Decision cohort: all 15 Stage-21 hard systems. There were no archived transition systems.
- Fixed evaluation gate: all three held-out RHSs converge and median iteration reduction is at least 40%; family gate is at least 12/15 systems.

## Evidence lineage

- Stage-19A coefficient rows and fixed nodes define every matrix.
- Stage-21 population results supply the already-validated strong-ILU baselines and held-out RHS protocol.
- `src/lcp_rbffd/experiments/stage22a_multi_rhs_functional_oracle.py` constructs and evaluates all new spaces.
- Per-system checkpoints under `systems/id_XX/` preserve raw solver and search evidence.

## Metric cautions

- Fourteen hard ILU baselines hit 300 iterations. Their iteration reductions and break-even estimates are censored by the established cap.
- Requested rank can exceed attainable numerical rank after pair preservation and redundancy removal; both are reported.
- Functional greedy uses two disjoint selection RHSs and is an expensive oracle diagnostic.
- Geometry pairs within the same system are dependent. Spearman coefficients are descriptive.

## Visual QA

- Six requested figures are static Matplotlib PNGs generated from validated CSVs.
- Equality and gate references are explicit, absolute-magnitude axes start at zero, and labels were inspected at full size.
"""
    (output / "source_notes.md").write_text(source_notes, encoding="utf-8")
    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    artifacts.append(
        {
            "path": str(report.relative_to(output.parents[1])),
            "bytes": report.stat().st_size,
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "external_report": True,
        }
    )
    _write_json(
        output / "manifest.json",
        {"stage": "stage22A_spatial_K_multi_rhs_functional_oracle", "artifact_count": len(artifacts), "artifacts": artifacts},
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "results-fd" / "stages" / "stage22A_spatial_K_multi_rhs_functional_oracle"
    report = root / "results" / "results-fd" / "reports" / "stage22A_spatial_K_multi_rhs_functional_oracle.md"
    settings = Stage22ASettings()
    solver = read_rows(output / "solver_results.csv")
    setup = read_rows(output / "setup_results.csv")
    summaries = read_rows(output / "system_rule_summary.csv")
    family = read_rows(output / "family_rule_summary.csv")
    geometry = read_rows(output / "geometry_functional_diagnostics.csv")
    correlations = read_rows(output / "geometry_functional_correlations.csv")
    singular = read_rows(output / "dictionary_singular_values.csv")
    rankings = system_ranking_diagnostics(geometry, output)
    diagnoses = diagnose_systems(summaries, singular, output)
    plot_results(output, summaries, geometry, family)
    write_report(report, output, summaries, family, geometry, correlations, rankings, diagnoses)
    # Run the focused Stage-22A tests and retain their exact output.
    completed = subprocess.run(
        ["/opt/miniconda3/bin/python3", "-m", "unittest", "tests.test_stage22a_multi_rhs_functional_oracle", "-v"],
        cwd=root,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )
    (output / "test_results.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    validate(output, report, settings, solver, setup, summaries, family)
    _write_json(
        output / "decision.json",
        {
            "decision": "A. Robust own-system oracle recovered; proceed to cross-system transfer.",
            "recommended_fixed_rule": "Z_union_n3_m128_q48",
            "family_gate_passed": True,
            "neural_training_performed": False,
            "cross_system_transfer_performed": False,
        },
    )
    write_supporting_files(output, report)
    print(report)


if __name__ == "__main__":
    main()
