from __future__ import annotations

import numpy as np
import scipy.linalg as la
import torch

from lcp_rbffd.experiments.stage17_solver_action_supervision import (
    ActionProbeBlock,
    Stage17Settings,
    action_error_from_coefficients,
    reduced_action_losses,
)


def _fixture() -> tuple[np.ndarray, ActionProbeBlock, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    n, p, q, k = 31, 9, 3, 11
    psi, _ = la.qr(rng.normal(size=(n, p)), mode="economic")
    matrix = rng.normal(size=(n, n)) + 4.0 * np.eye(n)
    coefficient = rng.normal(size=(p, q))
    oracle_z, _ = la.qr(rng.normal(size=(n, q)), mode="economic")
    oracle_w, oracle_r = la.qr(matrix @ oracle_z, mode="economic")
    oracle_u = la.solve_triangular(oracle_r.T, oracle_z.T, lower=True).T
    probes = rng.normal(size=(k, n))
    image = matrix @ psi
    b = probes @ oracle_w
    target = (oracle_u @ b.T).T
    block = ActionProbeBlock(
        ratio=1.0,
        split="test",
        gram_y=image.T @ image,
        residual_to_dictionary_image=probes @ image,
        residual_to_oracle_test=b,
        target_dictionary_coefficients=target @ psi,
        target_norm_squared=np.sum(target**2, axis=1),
        vector_norms=np.linalg.norm(probes, axis=1),
        trajectory_labels=np.repeat("fixture", k),
        iteration_indices=np.arange(k),
    )
    return coefficient, block, psi, matrix


def test_stage17_frozen_protocol() -> None:
    settings = Stage17Settings()
    assert settings.stage15.q == 8
    assert settings.stage15.hidden_widths == (64, 128)
    assert settings.stage15.model_seeds == (20261501, 20261502, 20261503)
    assert set(settings.training_probe_seeds).isdisjoint(settings.test_probe_seeds)


def test_reduced_action_error_matches_explicit_qr_action() -> None:
    coefficient, block, psi, matrix = _fixture()
    raw = psi @ coefficient
    z, _ = la.qr(raw, mode="economic")
    w, triangular = la.qr(matrix @ z, mode="economic")
    u = la.solve_triangular(triangular.T, z.T, lower=True).T
    # Recreate the fixture probes from the sufficient statistics: h=r.T A Psi.
    image = matrix @ psi
    probes = block.residual_to_dictionary_image @ np.linalg.pinv(image)
    predicted = (u @ (w.T @ probes.T)).T
    # The pseudoinverse reconstruction has the same Y.T r statistics, which is
    # all the candidate action uses.  Compare its norm/cross expansion.
    target_coeff = block.target_dictionary_coefficients
    predicted_coeff = predicted @ psi
    predicted_norm = np.sum(predicted_coeff**2, axis=1)
    cross = np.sum(predicted_coeff * target_coeff, axis=1)
    expected = np.sqrt(np.maximum(predicted_norm - 2.0 * cross + block.target_norm_squared, 0.0) / (block.target_norm_squared + 1.0e-30))
    actual = action_error_from_coefficients(coefficient, block, 1.0e-30)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-10)


def test_torch_action_loss_matches_numpy_error_squared() -> None:
    coefficient, block, _, _ = _fixture()
    actual = reduced_action_losses(
        torch.tensor(coefficient[None, :, :], dtype=torch.float64),
        [torch.tensor(block.gram_y, dtype=torch.float64)],
        [torch.tensor(block.residual_to_dictionary_image, dtype=torch.float64)],
        [torch.tensor(block.target_dictionary_coefficients, dtype=torch.float64)],
        [torch.tensor(block.target_norm_squared, dtype=torch.float64)],
        1.0e-30,
        0.0,
    ).detach().numpy()
    expected = action_error_from_coefficients(coefficient, block, 1.0e-30) ** 2
    np.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-10)
