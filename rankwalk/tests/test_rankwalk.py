import torch
from torch_geometric.data import Data
from rankwalk import RankWalkConv, RankWalkKernel

# Dummy graph
N = 10
F_in = 4
F_out = 8
x = torch.randn(N, F_in)
edge_index = torch.tensor([
    [0, 0, 1, 2, 2, 3, 4, 5, 6, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 7, 8]
], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

# Test RankWalkConv
conv = RankWalkConv(in_dim=F_in, out_dim=F_out, walk_length=3)
out = conv(data.x, data.edge_index)
print("RankWalkConv output shape:", out.shape)

# Test RankWalkKernel
kernel = RankWalkKernel(in_dim=F_in, walk_length=3)
tau = kernel(data.x, data.edge_index)
print("RankWalkKernel tau shape:", tau.shape)