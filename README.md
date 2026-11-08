# Hybrid Iterative Methods for RBF Linear Systems

Can machine learning capture what a strong classical preconditioner $M$ misses in RBF linear systems?
The object of study is the unresolved inverse action

$$d(r) = A^{-1}r - M^{-1}r,$$

used either as a learned direct correction or through a learned low-dimensional *difficult subspace*, which serves as a recycle or coarse space in GMRES.

## Reports

| | |
|---|---|
| [`reports/proposal.pdf`](reports/proposal.pdf) | Research proposal |
| [`reports/rbf-fd/main.pdf`](reports/rbf-fd/main.pdf) | **Sparse RBF-FD study** (final report, 25 pp.) |
| [`reports/global-rbf/main.pdf`](reports/global-rbf/main.pdf) | **Dense global RBF collocation study** (final report, 14 pp.) |

## Key results

**Sparse RBF-FD** (ILU + recycled GMRES)
- Strong ILU already handles Poisson, scalar-contrast, irregular-node and scale-only systems. Rotated anisotropic diffusion is the hard regime.
- Learned direct corrections never beat ILU. Learned subspace prediction only matches deterministic interpolation in a narrow, spatially constant coefficient family.
- With spatially varying tensor fields, deep own-system Arnoldi/Ritz spaces recover every one of the 15 hard fields. Rank adaptation was validated on 17 new systems. Cross-field reuse still fails.
- Target-local probe selection at rank 48 recovers most of the teacher benefit on 6 of 7 hard targets.
- In the remaining case, a rank-12 raw multi-RHS Krylov correction cuts the iteration count from **300 (cap) to 92**. Neither neural nor equivariant models reproduce this.
- **Open problem:** a neural selector using the coefficient field plus a very short local probe, which must beat classical selection at equal probe depth.

**Global RBF collocation** (dense LU / FGMRES)
- At moderate N, FP64 LU is fastest. FP32 pivoted LU is the best reusable iterative baseline.
- An exact harmonic-Ritz recycle space reduces iterations by **27.6%** on a difficult N = 400 system.
- Transfer to an adjacent shape parameter gives a median iteration ratio of about 1. Residual-history POD is not persistent. A shared offline basis needs 158 of 225 directions.
- So far, the evidence supports better numerical target design, not neural training.

## Code

```
src/lcp_rbffd/       RBF-FD assembly, ILU / smaller-stencil preconditioners, GMRES,
                     coarse spaces, Ritz/POD analysis, models, per-stage experiments
src/global_rbf/      global collocation, kernels, dense preconditioners, Krylov recycling, transfer
src/neurkitt_oracle/ GCRO-DR recycled GMRES (solver validation)
scripts/             run_* / validate_* entry points for each experiment
tests/               unit tests
```

```bash
pip install -e ".[ml]"
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python scripts/run_stages_1_5.py --output results/stages_1_5
```

Raw experiment outputs (~600 MB) are left out of this repository. Running the scripts regenerates them under `results/`.
