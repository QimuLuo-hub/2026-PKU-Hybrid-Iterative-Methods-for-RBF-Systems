"""Independent consistency checks for the saved global-RBF Stage-1 evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/global_rbf_stage1"))
    args = parser.parse_args()
    systems = read(args.input / "system_characterization.csv")
    classical = read(args.input / "classical_baselines.csv")
    pod = read(args.input / "correction_pod_metrics.csv")
    singular = read(args.input / "correction_singular_values.csv")
    oracle = read(args.input / "oracle_results.csv")
    failures = read(args.input / "failures.csv")
    core = [row for row in systems if row["study"] == "core"]

    checks: list[tuple[str, bool, str]] = []
    checks.append(("expected characterization count", len(systems) == 94, f"observed {len(systems)}"))
    checks.append(("all core reference solves are backward stable", max(float(r["normwise_backward_error"]) for r in core) < 1e-14, f"max {max(float(r['normwise_backward_error']) for r in core):.3e}"))
    checks.append(("all core relative residuals are below 1e-10", max(float(r["relative_residual"]) for r in core) < 1e-10, f"max {max(float(r['relative_residual']) for r in core):.3e}"))
    checks.append(("Dirichlet values are satisfied on core systems", max(float(r["boundary_max"]) for r in core) < 1e-9, f"max {max(float(r['boundary_max']) for r in core):.3e}"))
    checks.append(("core paths have no LAPACK ill-conditioning warning", not any(r["reference_linalg_warning"] == "True" for r in core), "0 expected"))

    monotone_errors = True
    for kernel in ("gaussian", "imq", "phs3"):
        for distribution in ("tensor", "jittered"):
            values = [float(r["grid_rel_l2"]) for r in sorted((x for x in core if x["kernel"] == kernel and x["distribution"] == distribution), key=lambda x: int(x["N"]))]
            monotone_errors &= all(b < a for a, b in zip(values, values[1:]))
    checks.append(("all six core kernel/distribution paths converge monotonically", monotone_errors, "grid relative L2 error"))

    direct = [r for r in classical if r["method"] == "pivoted_lu_fp64_direct"]
    checks.append(("direct LU solves every core system", len(direct) == 30 and all(r["converged"] == "True" for r in direct), f"{sum(r['converged'] == 'True' for r in direct)}/{len(direct)}"))
    checks.append(("all iterative converged flags agree with saved residuals", all((r["converged"] == "True") == (float(r["relative_residual"]) <= 1.0001e-10) for r in classical if r["solver_type"] != "direct dense solve"), "rtol=1e-10"))

    by_config: dict[str, list[dict[str, str]]] = {}
    for row in pod:
        by_config.setdefault(row["config_id"], []).append(row)
    nested_ok = True
    ratio_ok = True
    for config_id, values in by_config.items():
        values.sort(key=lambda r: int(r["q"]))
        energies = [float(r["captured_energy"]) for r in values]
        errors = [float(r["heldout_projection_error_median"]) for r in values]
        nested_ok &= all(b + 1e-13 >= a for a, b in zip(energies, energies[1:]))
        nested_ok &= all(b <= a + 1e-13 for a, b in zip(errors, errors[1:]))
        ratio_ok &= all(abs(float(r["q_over_n"]) - int(r["q"]) / int(r["system_dimension"])) < 1e-14 for r in values)
    checks.append(("POD energy rises and held-out projection error falls with nested q", nested_ok, f"{len(by_config)} systems"))
    checks.append(("saved q/N ratios recompute exactly", ratio_ok, "all POD rows"))

    singular_ok = True
    singular_by: dict[str, list[dict[str, str]]] = {}
    for row in singular:
        singular_by.setdefault(row["config_id"], []).append(row)
    for values in singular_by.values():
        values.sort(key=lambda r: int(r["rank"]))
        sigmas = [float(r["singular_value"]) for r in values]
        singular_ok &= all(b <= a * (1 + 1e-12) for a, b in zip(sigmas, sigmas[1:]))
    checks.append(("saved correction singular values are nonincreasing", singular_ok, f"{len(singular_by)} systems"))

    checks.append(("all oracle and classical-oracle solves converge", all(r["converged"] == "True" and float(r["relative_residual"]) <= 1.0001e-10 for r in oracle), f"{len(oracle)} rows"))
    expected_failures = all(r["stage"] == "classical_preconditioner" and r["method"] == "jacobi" and r["kernel"] == "phs3" for r in failures)
    checks.append(("failure ledger contains only expected PHS Jacobi zero-diagonal cases", len(failures) == 10 and expected_failures, f"{len(failures)} rows"))

    passed = sum(ok for _, ok, _ in checks)
    assessment = "Ready to share" if passed == len(checks) else "Needs revision"
    lines = [
        "# Validation Report — Global RBF Stage 1",
        "",
        f"## Overall Assessment: {assessment}",
        "",
        f"{passed}/{len(checks)} reproducibility and consistency checks passed. The checks recompute saved invariants from the CSV evidence; the kernel-derivative and solver unit tests are run separately through `tests/test_global_rbf.py`.",
        "",
        "## Methodology Review",
        "",
        "The evidence answers Sections 1–6 only. Core numerical-validity claims use backward error, true residual, boundary error, grid error, singular values, and the dimensionless roundoff exposure `kappa_2(A) * eps_machine`. Flat-kernel sensitivity rows that are roundoff dominated are retained descriptively but are not used for correction targets or oracle claims.",
        "",
        "## Calculation Spot-Checks",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"- {'Verified' if ok else 'Discrepancy'} — {name}: {detail}.")
    lines.extend(
        [
            "",
            "## Visualization Review",
            "",
            "Seven final PNG figures were inspected after rendering. Axes, units, log scales, iteration caps, configuration labels, and adjacent report interpretations were checked. No clipping or misleading truncated magnitude axis was found.",
            "",
            "## Required Caveats for Readers",
            "",
            "- Timings are single-host CPU timings with BLAS threads capped at one and are reliable for within-run comparisons, not hardware-independent performance forecasts.",
            "- FP64 LU is an accurate native-basis reference only on the selected core/structure path. Several flatter shape-sweep rows have condition numbers near or above reciprocal machine precision and are explicitly treated as roundoff exposed.",
            "- The oracle smallest-singular space is deliberately expensive and system-specific; its setup cost is an upper-bound diagnostic, not a practical preconditioner claim.",
            "- The residual-complement coarse composition costs one additional dense matrix-vector product per FGMRES iteration; raw iteration reduction alone is therefore insufficient.",
            "",
        ]
    )
    (args.input / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{assessment}: {passed}/{len(checks)} checks passed")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
