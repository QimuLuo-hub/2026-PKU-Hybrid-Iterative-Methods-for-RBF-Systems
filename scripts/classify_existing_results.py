#!/usr/bin/env python3
"""Classify existing immutable experiment artifacts by RBF branch.

This script copies, never transforms, experiment outputs into the two branch
archives requested for the publication reports.  Source paths remain in place
so that historical scripts and report links continue to work.
"""

from __future__ import annotations

import hashlib
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results"


@dataclass(frozen=True)
class Source:
    path: str
    destination: str
    stage: str
    description: str


FD_SOURCES = [
    Source("results/stages_1_5", "stages/stages_1_5", "Stages 1--5", "RBF-FD discretization, ILU/smaller-stencil baselines, correction snapshots, POD diagnostics, and coarse-space experiments."),
    Source("results/stage6_harder_regimes_and_direct_ml", "stages/stage6_harder_regimes_and_direct_ml", "Stage 6", "Harder sparse RBF-FD regimes, ILU budget study, correction POD, transfer diagnostics, and first direct MLP baseline."),
    Source("results/stage7_solver_aware_mlp_vs_gnn", "stages/stage7_solver_aware_mlp_vs_gnn", "Stage 7", "Solver-aware direct-complement MLP/GNN comparison, normalization diagnostics, permutations, and POD target stability."),
    Source("results/stage8_direct_onpolicy_and_subspace_stability", "stages/stage8_direct_onpolicy_and_subspace_stability", "Stage 8", "On-policy direct-complement aggregation, damping, and classical difficult-space stability/oracle tests."),
    Source("results/stage9_hard_regime_discovery_and_scalability", "stages/stage9_hard_regime_discovery_and_scalability", "Stage 9", "RBF-FD scale, coefficient, geometry, anisotropy, repeated-solve, correction-structure, and oracle-recycling results."),
    Source("results/stage10_ritz_transfer_and_mode_tracking", "stages/stage10_ritz_transfer_and_mode_tracking", "Stage 10", "Rotated-anisotropy Ritz dimension, transfer, mode-tracking, and robustness experiments."),
    Source("results/stage11_ritz_cluster_stability", "stages/stage11_ritz_cluster_stability", "Stage 11", "Ritz cluster/invariant-block stability, criticality, and transfer experiments."),
    Source("results/stage12_nonsymmetric_petrov_galerkin_recycling_validation", "stages/stage12_nonsymmetric_petrov_galerkin_recycling_validation", "Stage 12", "Nonsymmetry diagnostics and Galerkin, Petrov--Galerkin, and recycling coupling validation."),
    Source("results/stage13_persistent_preconditioner_vs_krylov_recycling", "stages/stage13_persistent_preconditioner_vs_krylov_recycling", "Stage 13", "Fair persistent two-level versus recycled-GMRES integration experiment using a common Ritz space."),
    Source("results/stage14_solver_aware_target_loss_diagnostics", "stages/stage14_solver_aware_target_loss_diagnostics", "Stage 14", "Candidate-space geometry and short-rollout metrics against recycled-GMRES performance."),
    Source("results/stage15_family_specific_W_target_MLP", "stages/stage15_family_specific_W_target_MLP", "Stage 15", "Scalar-anisotropy family-specific reduced-dictionary MLP and deterministic baselines."),
    Source("results/stage16_W_aware_dictionary_ablation", "stages/stage16_W_aware_dictionary_ablation", "Stage 16", "Z-POD, AZ-aware, and joint dictionary representation ablation."),
    Source("results/stage17_solver_action_supervision", "stages/stage17_solver_action_supervision", "Stage 17", "Solver-action supervision ablation with fixed dictionary and MLP."),
    Source("results/stage18_two_parameter_anisotropy_family", "stages/stage18_two_parameter_anisotropy_family", "Stage 18", "Spatially constant two-parameter anisotropy family, Grassmann interpolation, and MLP prediction."),
    Source("results/stage19A_spatial_coefficient_oracle_validation", "stages/stage19A_spatial_coefficient_oracle_validation", "Stage 19A", "Spatially varying tensor-field hardness scan, q=8 oracle validation, and oracle-space complexity."),
    Source("results/stage19C_spatial_coefficient_oracle_mechanism_diagnostic", "stages/stage19C_spatial_coefficient_oracle_mechanism_diagnostic", "Stage 19C", "Spatially varying tensor-field oracle q-sensitivity, ordinary/harmonic Ritz, and correction-POD diagnostics."),
    Source("experiments/results", "preliminary/experiments_results", "Preliminary RBF-FD experiment", "Full preliminary local PHS RBF-FD ILU, correction-POD, and coarse-correction outputs."),
    Source("experiments/results_quick", "preliminary/experiments_results_quick", "Preliminary RBF-FD quick run", "Quick local RBF-FD baseline and POD outputs."),
    Source("experiments/results_quick_coarse_first", "preliminary/experiments_results_quick_coarse_first", "Preliminary RBF-FD ordering check", "Quick coarse-first RBF-FD solve outputs."),
    Source("experiments/results_quick_galerkin", "preliminary/experiments_results_quick_galerkin", "Preliminary RBF-FD Galerkin check", "Quick Galerkin-style RBF-FD coarse-correction outputs."),
    Source("experiments/results_quick_weighted", "preliminary/experiments_results_quick_weighted", "Preliminary RBF-FD weighted check", "Quick weighted RBF-FD coarse-space outputs."),
    Source("experiments/results_seed_robustness", "preliminary/experiments_results_seed_robustness", "Preliminary RBF-FD robustness", "RBF-FD random-seed robustness outputs."),
    Source("rbf_learned_preconditioner_experiment_report.md", "reports/rbf_learned_preconditioner_experiment_report.md", "Preliminary RBF-FD report", "Narrative report for the preliminary local RBF-FD experiments."),
    Source("Users/daisy/Desktop/iteration/rbf_learned_preconditioner_experiment_report.md", "reports/legacy_nested_rbf_learned_preconditioner_experiment_report.md", "Preliminary RBF-FD report", "Legacy duplicate of the preliminary local RBF-FD experiment report found in a nested path."),
    Source("stage19A_spatial_coefficient_oracle_validation.md", "reports/stage19A_spatial_coefficient_oracle_validation.md", "Stage 19A", "Source report for spatially varying coefficient oracle validation."),
    Source("stage19C_spatial_coefficient_oracle_mechanism_diagnostic.md", "reports/stage19C_spatial_coefficient_oracle_mechanism_diagnostic.md", "Stage 19C", "Source report for the spatially varying coefficient oracle mechanism diagnostic."),
    Source("tmp/stage13_smoke", "verification/stage13_smoke", "Stage 13 verification", "First saved smoke-test output for the persistent-versus-recycling implementation."),
    Source("tmp/stage13_smoke2", "verification/stage13_smoke2", "Stage 13 verification", "Second saved smoke-test output for the persistent-versus-recycling implementation."),
    Source("tmp/stage13_smoke3", "verification/stage13_smoke3", "Stage 13 verification", "Third saved smoke-test output for the persistent-versus-recycling implementation."),
    Source("results/neurkitt_oracle", "supporting_solver_validation/neurkitt_oracle", "External solver validation", "Five-point Darcy/NeurKItt oracle outputs used only to validate the recycle-space solver; not an RBF-FD numerical result."),
    Source("results/neurkitt_recycle_space", "supporting_solver_validation/neurkitt_recycle_space", "External solver validation", "Five-point Darcy/NeurKItt recycle-space outputs used only to validate the recycle-space solver; not an RBF-FD numerical result."),
]


GLOBAL_RESULT_DIRS = [
    "global_rbf_classical_preconditioners",
    "global_rbf_classical_preconditioners_quick",
    "global_rbf_classical_scaling",
    "global_rbf_classical_scaling_smoke",
    "global_rbf_epsilon_recycle_transfer",
    "global_rbf_exact_ZS_coarse_integrations",
    "global_rbf_exact_ZS_coarse_integrations_quick",
    "global_rbf_fair_targets_rank_transfer",
    "global_rbf_fair_targets_rank_transfer_quick",
    "global_rbf_harmonic_cluster_diagnostic",
    "global_rbf_krylov_oracle_spaces",
    "global_rbf_krylov_oracle_spaces_quick",
    "global_rbf_krylov_recycle_largeN",
    "global_rbf_krylov_recycle_n400",
    "global_rbf_oracle_target_selection",
    "global_rbf_oracle_target_selection_quick",
    "global_rbf_recycle_benchmark_search",
    "global_rbf_recycle_benchmark_search_ls81",
    "global_rbf_residual_pod_stage1",
    "global_rbf_stage1",
    "global_rbf_stage1_quick",
    "global_rbf_stage2",
    "global_rbf_stage2_quick",
    "global_rbf_subspace_dataset",
    "global_rbf_subspace_dataset_quick",
]

GLOBAL_SOURCES = [
    *[
        Source(
            f"results/{name}",
            f"experiments/{name}",
            name.replace("global_rbf_", "Global RBF ").replace("_", " "),
            "Dense global-RBF/Kansa experimental outputs; see the contained settings, manifests, and source report for exact provenance.",
        )
        for name in GLOBAL_RESULT_DIRS
    ],
    Source("frequency-test", "frequency_diagnostics", "Global RBF frequency diagnostics", "Dense one-, two-, and three-dimensional global-RBF spectral/frequency plots and CSV summaries, including a finite-difference control."),
    Source("report-global", "publication_report", "Global RBF publication report", "Existing Global RBF LaTeX report, tables, figures, bibliography, and compiled PDF."),
]

for report in sorted(ROOT.glob("global_rbf_*report*.md")) + sorted(ROOT.glob("global_rbf_*report*.html")):
    GLOBAL_SOURCES.append(
        Source(
            str(report.relative_to(ROOT)),
            f"reports/{report.name}",
            "Global RBF source report",
            "Source report for dense global-RBF/Kansa experiments.",
        )
    )


def files_under(path: Path):
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name == ".DS_Store" or "__pycache__" in candidate.parts:
            continue
        # The frequency directory also contains executable experiment scripts.
        if path.name == "frequency-test" and candidate.suffix == ".py":
            continue
        yield candidate


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def classify(branch: str, sources: list[Source]) -> tuple[int, int]:
    target_root = ARCHIVE / f"results-{branch}"
    target_root.mkdir(parents=True, exist_ok=True)
    rows = []
    total_bytes = 0
    seen_destinations: set[Path] = set()

    for source in sources:
        origin = ROOT / source.path
        if not origin.exists():
            raise FileNotFoundError(origin)
        for item in files_under(origin):
            relative = Path(item.name) if origin.is_file() else item.relative_to(origin)
            destination = target_root / source.destination / relative
            if destination in seen_destinations:
                raise RuntimeError(f"duplicate destination: {destination}")
            seen_destinations.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            source_hash = digest(item)
            copied_hash = digest(destination)
            if source_hash != copied_hash:
                raise RuntimeError(f"checksum mismatch: {item} -> {destination}")
            size = item.stat().st_size
            total_bytes += size
            rows.append(
                (
                    item.name,
                    item.relative_to(ROOT).as_posix(),
                    destination.relative_to(ROOT).as_posix(),
                    "RBF-FD" if branch == "fd" else "Global RBF",
                    source.stage,
                    source.description,
                    size,
                    source_hash,
                )
            )

    title = "RBF-FD result manifest" if branch == "fd" else "Global RBF result manifest"
    note = (
        "The `supporting_solver_validation` subtree is retained with the RBF-FD research branch because it validates the recycled-GMRES implementation used by that branch. Its Darcy five-point results are explicitly not treated as RBF-FD numerical evidence in the publication report."
        if branch == "fd"
        else "The finite-difference control figure in `frequency_diagnostics` is an internal control from a Global RBF frequency experiment, not an RBF-FD branch result."
    )
    lines = [
        f"# {title}",
        "",
        "This archive is a byte-for-byte copy of existing artifacts. No experiment was rerun and no numerical value was transformed. Original files remain in place for reproducibility.",
        "",
        note,
        "",
        f"- Classified files: {len(rows)}",
        f"- Classified bytes: {total_bytes}",
        "- Integrity check: SHA-256 matched after every copy",
        "",
        "| Filename | Original location | New location | Branch | Source stage/experiment | Short description | Bytes | SHA-256 |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    (target_root / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows), total_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleanup-sources",
        action="store_true",
        help="remove verified redundant sources after classification",
    )
    args = parser.parse_args()
    fd_count, fd_bytes = classify("fd", FD_SOURCES)
    global_count, global_bytes = classify("global", GLOBAL_SOURCES)
    print(f"RBF-FD: {fd_count} files, {fd_bytes} bytes")
    print(f"Global RBF: {global_count} files, {global_bytes} bytes")
    if args.cleanup_sources:
        cleanup_sources()


def cleanup_sources() -> None:
    """Delete only originals whose classified copies still match exactly."""

    skipped = {ROOT / "report-global"}
    removed_files = 0
    removed_bytes = 0
    sources = [("fd", source) for source in FD_SOURCES] + [
        ("global", source) for source in GLOBAL_SOURCES
    ]
    for branch, source in sources:
        origin = ROOT / source.path
        if not origin.exists() or origin in skipped:
            continue
        target_root = ARCHIVE / f"results-{branch}" / source.destination
        classified = list(files_under(origin))
        for item in classified:
            relative = Path(item.name) if origin.is_file() else item.relative_to(origin)
            destination = target_root / relative
            if not destination.is_file() or digest(item) != digest(destination):
                raise RuntimeError(
                    f"refusing cleanup because archive verification failed: {item}"
                )
            removed_files += 1
            removed_bytes += item.stat().st_size

        if origin.is_file():
            origin.unlink()
        elif origin.name == "frequency-test":
            # Keep the experiment scripts; remove only the archived outputs.
            for item in classified:
                item.unlink()
        else:
            shutil.rmtree(origin)

    # Disposable QA renders, smoke copies, and filesystem caches are not
    # scientific evidence.  The smoke outputs themselves were archived above.
    disposable = [ROOT / "tmp"]
    for path in disposable:
        if path.exists():
            shutil.rmtree(path)
    for path in sorted(ROOT.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
    for path in ROOT.rglob(".DS_Store"):
        path.unlink()
    # The legacy nested report path becomes empty after its archived source is
    # removed.  Prune empty directories without touching nonempty user data.
    legacy_root = ROOT / "Users"
    if legacy_root.exists():
        for path in sorted(legacy_root.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if not any(legacy_root.iterdir()):
            legacy_root.rmdir()
    for branch in ("fd", "global"):
        manifest = ARCHIVE / f"results-{branch}" / "manifest.md"
        contents = manifest.read_text(encoding="utf-8")
        contents = contents.replace(
            "Original files remain in place for reproducibility.",
            "The manifest records historical source locations. Redundant originals were removed only after checksum verification; the classified copies are the retained evidence archive.",
        )
        manifest.write_text(contents, encoding="utf-8")
    print(f"Removed redundant originals: {removed_files} files, {removed_bytes} bytes")


if __name__ == "__main__":
    main()
