import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernel import RankWalkKernel

class RankWalkConv(nn.Module):
    """
    RankWalk-GNN convolution layer (PyTorch Geometric)
    """

    def __init__(self, in_dim: int, out_dim: int, walk_length: int = 5, beta: float = 0.1):
        super().__init__()
        self.kernel = RankWalkKernel(in_dim=in_dim, walk_length=walk_length, beta=beta)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        """
        x: (N, d)
        edge_index: (2, E)
        Returns:
            out: (N, out_dim)
        """
        tau = self.kernel(x, edge_index)  # (N, N)

        # Convert to soft ranks
        diff = tau.unsqueeze(2) - tau.unsqueeze(1)
        ranks = torch.sigmoid(diff / self.kernel.beta).sum(dim=2)

        # Rank-based affinity
        A = torch.exp(-ranks)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-6)

        # Message aggregation
        out = A @ x
        return self.lin(out)