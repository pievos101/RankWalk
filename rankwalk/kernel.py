import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class RankWalkKernel(MessagePassing):
    """
    Differentiable RankWalk kernel (PyG-compatible)
    Computes soft node-to-node arrival / affinity matrix
    """

    def __init__(self, in_dim: int, walk_length: int = 5, beta: float = 0.1, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.walk_length = walk_length
        self.beta = beta

        # Linear layer to compute feature-based similarity for biasing walks
        self.lin = nn.Linear(in_dim, in_dim, bias=False)

        # Depth weights for combining multiple walk steps
        self.alpha = nn.Parameter(torch.ones(walk_length + 1))

    def forward(self, x, edge_index):
        """
        x: (N, d) node features
        edge_index: (2, E) edge list
        Returns:
            tau: (N, N) soft node-to-node affinity matrix
        """
        N = x.size(0)
        z = torch.sigmoid(self.lin(x))  # feature-based node embeddings

        # Initialize identity matrix: start at each node
        q = torch.eye(N, device=x.device)
        qs = [q]

        # Perform walk_length steps
        for _ in range(self.walk_length):
            q = self.propagate(edge_index, x=z, q=q)
            # row-normalize
            q = q / (q.sum(dim=1, keepdim=True) + 1e-6)
            qs.append(q)

        qs = torch.stack(qs, dim=1)  # (N, T+1, N)
        alpha = F.softmax(self.alpha, dim=0)
        tau = (alpha.view(1, -1, 1) * qs).sum(dim=1)

        return tau  # (N, N)

    def message(self, q_j, x_i, x_j):
        """
        Message function for PyG MessagePassing
        q_j: (E, N)
        x_i, x_j: (E, d)
        """
        sim = (x_i * x_j).sum(dim=-1, keepdim=True)
        return sim * q_j