#!/usr/bin/env python3
"""Separate Arnoldi-depth and recycle-rank effects in Stage 20."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lcp_rbffd.experiments.stage19a_spatial_coefficient_oracle import _discrete_rhs
from lcp_rbffd.experiments.stage19c_oracle_mechanism import _actual_prefix
from lcp_rbffd.experiments.stage20_spatial_k_robust_oracle import (
    CHARCOAL,
    LIGHT_GRAY,
    Candidate,
    Stage20Settings,
    _evaluate_basis,
    _harmonic_from_arnoldi,
    _load_context,
    _ordinary_family,
    _orthonormalize,
    _read_rows,
)
from lcp_rbffd.experiments.stage9_scalability import _write_csv


def run(root: Path) -> Path:
    settings = Stage20Settings()
    fd_root = root / "results" / "results-fd"
    stage19a = fd_root / "stages" / "stage19A_spatial_coefficient_oracle_validation"
    stage19c = fd_root / "stages" / "stage19C_spatial_coefficient_oracle_mechanism_diagnostic"
    output = fd_root / "stages" / "stage20_spatial_K_robust_oracle_search"
    node_data = np.load(stage19a / "fixed_nodes.npz")
    coefficient_rows = {
        int(row["sample_id"]): row
        for row in _read_rows(stage19a / "coefficient_samples.csv")
    }
    baseline_rows = {
        (int(row["sample_id"]), int(row["rhs_seed"])): int(row["iterations"])
        for row in _read_rows(stage19c / "baseline_results.csv")
    }
    records = []
    for ordinal, sample_id in enumerate(settings.sample_ids, start=1):
        print(f"[Stage20 rank ablation {ordinal}/{len(settings.sample_ids)}] ID {sample_id}", flush=True)
        context = _load_context(
            root,
            sample_id,
            coefficient_rows,
            node_data["nodes"],
            node_data["boundary"],
            settings,
        )
        baseline = {
            seed: baseline_rows[(sample_id, seed)] for seed in settings.heldout_rhs_seeds
        }
        archived = np.load(
            stage19c / "spaces" / f"sample_{sample_id:02d}_diagnostic_spaces.npz"
        )
        construction_rhs, _ = _discrete_rhs(context.system, settings.construction_rhs_seed)
        ordinary, arnoldi = _ordinary_family(
            context.system.matrix,
            context.ilu,
            construction_rhs,
            max_q=64,
            arnoldi_dimension=settings.long_arnoldi_dimension,
            seed=settings.seed + sample_id,
        )
        harmonic = _harmonic_from_arnoldi(context.system.matrix, arnoldi, 64)
        candidates = []
        for family_name, key in (("ordinary", "ordinary_ritz"), ("harmonic", "harmonic_ritz")):
            candidates.append(
                Candidate(
                    method=f"{family_name}_m64_q32",
                    label=f"{family_name}_m64_q32",
                    basis=_orthonormalize(np.asarray(archived[key][:, :32], dtype=float)),
                    construction_seconds=0.0,
                    metadata={"arnoldi_dimension": 64, "requested_q": 32},
                )
            )
        for family in (ordinary, harmonic):
            for q in settings.q_values:
                basis, actual_q = _actual_prefix(family.basis, family.blocks, q)
                candidates.append(
                    Candidate(
                        method=f"{family.name}_m128_q{q}",
                        label=f"{family.name}_m128_q{actual_q}",
                        basis=basis,
                        construction_seconds=family.construction_seconds,
                        metadata={"arnoldi_dimension": 128, "requested_q": q},
                    )
                )
        for candidate in candidates:
            rows, _ = _evaluate_basis(
                context,
                candidate,
                settings.heldout_rhs_seeds,
                settings.solver,
                "heldout_rank_depth_ablation",
                baseline,
            )
            records.extend(rows)

    path = output / "arnoldi_rank_ablation.csv"
    _write_csv(path, records)
    labels = (
        "ordinary_m64_q32",
        "ordinary_m128_q32",
        "ordinary_m128_q64",
        "harmonic_m64_q32",
        "harmonic_m128_q32",
        "harmonic_m128_q64",
    )
    medians = {
        (sample_id, label): float(
            np.median(
                [
                    int(row["iterations"])
                    for row in records
                    if int(row["sample_id"]) == sample_id and row["method"] == label
                ]
            )
        )
        for sample_id in settings.sample_ids
        for label in labels
    }
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharex=True, sharey=True)
    for axis, sample_id in zip(axes.flat, settings.sample_ids):
        values = [medians[(sample_id, label)] for label in labels]
        axis.plot(range(len(labels)), values, marker="o", color="#2864B7")
        axis.axhline(300, color="#D97706", linestyle="--", linewidth=0.8)
        axis.set_title(f"ID {sample_id}")
        axis.grid(color=LIGHT_GRAY, linewidth=0.6)
        axis.set_xticks(range(len(labels)), ["O64/32", "O128/32", "O128/64", "H64/32", "H128/32", "H128/64"], rotation=45, ha="right")
    axes[0, 0].set_ylabel("Median iterations")
    axes[1, 0].set_ylabel("Median iterations")
    fig.suptitle("Arnoldi-depth and recycle-rank ablation\nO/H = ordinary/harmonic, construction depth/rank", color=CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output / "plots" / "arnoldi_rank_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path, flush=True)
    return path


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
