# test_topk_rankwalk.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from torch_geometric.datasets import Planetoid
import numpy as np

# -------------------------------
# Soft TopK RankWalk Convolution
# -------------------------------
class SoftTopKRankWalk(nn.Module):
    def __init__(self, in_dim, out_dim, walk_length=5, topk=10):
        super().__init__()
        self.walk_length = walk_length
        self.topk = topk
        self.lin = nn.Linear(in_dim, out_dim)
        self.node_proj = nn.Linear(in_dim, in_dim)  # feature projection for attention

    def forward(self, x, edge_index):
        N = x.size(0)
        h = x

        # Build adjacency dictionary
        adj_dict = {i: [] for i in range(N)}
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            adj_dict[s].append(d)

        # Soft walk aggregation
        h_walk = h
        for _ in range(self.walk_length):
            h_new = torch.zeros_like(h_walk)
            for i in range(N):
                neighbors = adj_dict[i]
                if not neighbors:
                    h_new[i] = h_walk[i]
                    continue
                neigh_feat = h_walk[neighbors]  # (k, d)
                proj_i = self.node_proj(h_walk[i]).unsqueeze(0)  # (1, d)
                sim = (proj_i * neigh_feat).sum(dim=-1)  # cosine-like
                attn = F.softmax(sim, dim=0)
                h_new[i] = (attn.unsqueeze(1) * neigh_feat).sum(dim=0)
            h_walk = h_new
        return self.lin(h_walk)

# -------------------------------
# InfoNCE-style contrastive loss
# -------------------------------
def contrastive_infomax(h, y, tau=0.5):
    h = F.normalize(h, dim=1)
    sim_matrix = h @ h.t()  # cosine similarity
    N = h.size(0)
    y = y.view(-1)
    loss = 0.0
    for i in range(N):
        pos_idx = (y == y[i]).nonzero(as_tuple=False).squeeze()
        neg_idx = (y != y[i]).nonzero(as_tuple=False).squeeze()
        pos_sim = sim_matrix[i, pos_idx]
        neg_sim = sim_matrix[i, neg_idx]
        numerator = torch.exp(pos_sim / tau).sum()
        denominator = numerator + torch.exp(neg_sim / tau).sum()
        loss += -torch.log(numerator / (denominator + 1e-8))
    loss /= N
    return loss

# -------------------------------
# Load dataset
# -------------------------------
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
x, y = data.x, data.y
edge_index = data.edge_index

# Downsample for quick testing
num_nodes = 500
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
model = SoftTopKRankWalk(in_dim=x.size(1), out_dim=64, walk_length=5, topk=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training loop
# -------------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    embeddings = model(x, edge_index_sub)
    loss = contrastive_infomax(embeddings, y)
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

# Ward (AgglomerativeClustering)
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward').fit(embeddings)
y_pred_w = ward.labels_

print("\nContrastive TopK RankWalk -> KMeans ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_k),
    NMI(y.numpy(), y_pred_k)
))
print("Contrastive TopK RankWalk -> Ward ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_w),
    NMI(y.numpy(), y_pred_w)
))