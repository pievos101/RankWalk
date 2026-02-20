# test_unsupervised.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from torch_geometric.datasets import Planetoid
import numpy as np

# -------------------------------
# RankWalk Convolution (scatter-free)
# -------------------------------
class RankWalkConv(nn.Module):
    def __init__(self, in_dim, out_dim, walk_length=20):
        super().__init__()
        self.walk_length = walk_length
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = x

        # Build adjacency matrix (dense)
        adj = torch.zeros(N, N, device=x.device)
        src, dst = edge_index
        adj[src, dst] = 1.0

        # Row-normalize
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)

        # Walk aggregation
        for _ in range(self.walk_length):
            h = adj @ h

        # Linear transformation
        h = self.lin(h)
        return h  # per-node embeddings

# -------------------------------
# Unsupervised Contrastive Loss
# -------------------------------
def contrastive_loss_unsupervised(h, edge_index, margin=0.5):
    """
    Positive pairs: neighbors in edge_index
    Negative pairs: all others
    """
    N = h.size(0)
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)  # (N, N)

    # Positive mask
    pos_mask = torch.zeros(N, N, device=h.device)
    src, dst = edge_index
    pos_mask[src, dst] = 1.0
    pos_mask[torch.arange(N), torch.arange(N)] = 1.0  # self-positive

    neg_mask = 1 - pos_mask

    loss = (pos_mask * (1 - sim) + neg_mask * F.relu(sim - margin)).mean()
    return loss

# -------------------------------
# Load dataset
# -------------------------------
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
x, y = data.x, data.y
edge_index = data.edge_index

# Downsample for quick testing
num_nodes = 1000
idx = np.random.choice(x.size(0), num_nodes, replace=False)
x = x[idx]
y = y[idx]

# Map edges to subsampled nodes
mask = np.isin(edge_index[0].numpy(), idx) & np.isin(edge_index[1].numpy(), idx)
edge_index_sub = edge_index[:, mask]
id_map = {old: new for new, old in enumerate(idx)}
edge_index_sub = torch.tensor([[id_map[int(i)] for i in edge_index_sub[0]],
                               [id_map[int(i)] for i in edge_index_sub[1]]], dtype=torch.long)

print(f"Downsampled graph: {x.size(0)} nodes, {edge_index_sub.size(1)} edges")
print(f"Number of true classes: {len(torch.unique(y))}")

# -------------------------------
# Model and optimizer
# -------------------------------
model = RankWalkConv(in_dim=x.size(1), out_dim=64, walk_length=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training loop
# -------------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    embeddings = model(x, edge_index_sub)
    loss = contrastive_loss_unsupervised(embeddings, edge_index_sub)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Clustering evaluation
# -------------------------------
with torch.no_grad():
    embeddings = model(x, edge_index_sub).numpy()

# KMeans
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
y_pred_k = kmeans.labels_

# Ward (Agglomerative)
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward').fit(embeddings)
y_pred_w = ward.labels_

print("\nContrastive RankWalk -> KMeans ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_k),
    NMI(y.numpy(), y_pred_k)
))

print("Contrastive RankWalk -> Ward ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_w),
    NMI(y.numpy(), y_pred_w)
))