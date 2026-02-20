import torch
import torch.nn as nn
from .kernel import RankWalkKernel

class RankWalkConv(nn.Module):
    """
    RankWalk convolution using tau affinity (scatter-free)
    """

    def __init__(self, in_dim, out_dim, walk_length=5, beta=0.1):
        super().__init__()
        self.kernel = RankWalkKernel(in_dim, walk_length=walk_length, beta=beta)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        tau = self.kernel(x, edge_index)  # (N, N)

        # Soft ranks
        diff = tau.unsqueeze(2) - tau.unsqueeze(1)
        ranks = torch.sigmoid(diff / self.kernel.beta).sum(dim=2)

        # Affinity matrix
        A = torch.exp(-ranks)
        A = A / (A.sum(dim=1, keepdim=True) + 1e-6)

        out = A @ x
        return self.lin(out)