#!/usr/bin/env python3
"""Test geometric and functional stability of Stage-20 m=128 spaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lcp_rbffd.experiments.stage19a_spatial_coefficient_oracle import _discrete_rhs
from lcp_rbffd.experiments.stage19c_oracle_mechanism import _actual_prefix
from lcp_rbffd.experiments.stage20_spatial_k_robust_oracle import (
    Candidate,
    Stage20Settings,
    _evaluate_basis,
    _harmonic_from_arnoldi,
    _load_context,
    _ordinary_family,
    _read_rows,
)
from lcp_rbffd.experiments.stage9_scalability import _write_csv
from lcp_rbffd.subspace_analysis import generalized_subspace_metrics


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
    construction_seeds = (settings.construction_rhs_seed,) + settings.stability_rhs_seeds
    rows = []
    for ordinal, sample_id in enumerate(settings.sample_ids, start=1):
        print(f"[Stage20 construction stability {ordinal}/{len(settings.sample_ids)}] ID {sample_id}", flush=True)
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
        families = {}
        for seed in construction_seeds:
            rhs, _ = _discrete_rhs(context.system, seed)
            ordinary, arnoldi = _ordinary_family(
                context.system.matrix,
                context.ilu,
                rhs,
                max_q=64,
                arnoldi_dimension=128,
                seed=settings.seed + sample_id + seed,
            )
            families[(seed, "ordinary")] = ordinary
            families[(seed, "harmonic")] = _harmonic_from_arnoldi(
                context.system.matrix, arnoldi, 64
            )
        for family_name in ("ordinary", "harmonic"):
            for q in (24, 64):
                primary_family = families[(construction_seeds[0], family_name)]
                primary_basis, primary_q = _actual_prefix(
                    primary_family.basis, primary_family.blocks, q
                )
                for seed in construction_seeds:
                    family = families[(seed, family_name)]
                    basis, actual_q = _actual_prefix(family.basis, family.blocks, q)
                    geometry = generalized_subspace_metrics(primary_basis, basis)
                    candidate = Candidate(
                        method=f"{family_name}_m128_q{q}_construction_stability",
                        label=f"{family_name}_m128_q{actual_q}_seed{seed}",
                        basis=basis,
                        construction_seconds=family.construction_seconds,
                        metadata={
                            "construction_rhs_seed": seed,
                            "reference_construction_rhs_seed": construction_seeds[0],
                            "requested_q": q,
                            "reference_q": primary_q,
                            **geometry,
                        },
                    )
                    evaluation, _ = _evaluate_basis(
                        context,
                        candidate,
                        settings.heldout_rhs_seeds,
                        settings.solver,
                        "construction_stability",
                        baseline,
                    )
                    rows.extend(evaluation)
    path = output / "m128_construction_stability.csv"
    _write_csv(path, rows)
    print(path, flush=True)
    return path


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
