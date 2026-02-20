import torch
import torch.nn as nn
import torch.nn.functional as F

class RankWalkKernel(nn.Module):
    """
    Differentiable RankWalk kernel without torch_scatter.
    Computes soft node-to-node affinity matrix.
    """

    def __init__(self, in_dim: int, walk_length: int = 5, beta: float = 0.1):
        super().__init__()
        self.walk_length = walk_length
        self.beta = beta
        self.lin = nn.Linear(in_dim, in_dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(walk_length + 1))

    def forward(self, x, edge_index):
        """
        x: (N, d)
        edge_index: (2, E)
        Returns:
            tau: (N, N) soft node-to-node affinity
        """
        N = x.size(0)
        device = x.device
        z = torch.sigmoid(self.lin(x))

        # Start node anchored: identity
        q = torch.eye(N, device=device)
        qs = [q]

        # Precompute adjacency as dense for matrix multiplication
        adj = torch.zeros(N, N, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1.0
        adj = adj / deg  # row-normalized

        for _ in range(self.walk_length):
            q = adj @ q  # propagate probability
            qs.append(q)

        qs = torch.stack(qs, dim=1)  # (N, T+1, N)
        alpha = F.softmax(self.alpha, dim=0)
        tau = (alpha.view(1, -1, 1) * qs).sum(dim=1)  # weighted sum over walk steps
        return tau