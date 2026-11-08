"""Fast structural tests for the message-passing direct complement."""

import unittest

import numpy as np
import torch

from lcp_rbffd.models.direct_gnn import (
    DirectComplementGNN,
    DirectGNNConfig,
)


class DirectGNNTests(unittest.TestCase):
    def test_message_passing_is_permutation_equivariant(self) -> None:
        torch.manual_seed(5)
        model = DirectComplementGNN(
            node_feature_width=4,
            edge_feature_width=3,
            config=DirectGNNConfig(hidden_width=8, message_layers=2),
        ).eval()
        nodes = torch.randn(2, 5, 4)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 1, 2], [0, 1, 2, 3, 4, 2, 1]],
            dtype=torch.long,
        )
        edges = torch.randn(edge_index.shape[1], 3)
        degree = torch.bincount(edge_index[1], minlength=5).float()
        permutation = np.array([3, 0, 4, 1, 2])
        old_to_new = np.argsort(permutation)
        permuted_index = torch.from_numpy(old_to_new[edge_index.numpy()]).long()
        permuted_nodes = nodes[:, permutation]
        permuted_degree = degree[permutation]
        with torch.no_grad():
            original = model(nodes, edge_index, edges, degree)
            permuted = model(
                permuted_nodes,
                permuted_index,
                edges,
                permuted_degree,
            )
        np.testing.assert_allclose(
            permuted.numpy(), original.numpy()[:, permutation], rtol=2.0e-6, atol=2.0e-6
        )


if __name__ == "__main__":
    unittest.main()
