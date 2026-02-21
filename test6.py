# test_unsupervised_topk_rankwalk.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from torch_geometric.datasets import Planetoid
import numpy as np

# -------------------------------
# TopK-anchored RankWalk Convolution (scatter-free)
# -------------------------------
class TopKRankWalkConv(nn.Module):
    def __init__(self, in_dim, out_dim, walk_length=5, topk=5):
        super().__init__()
        self.walk_length = walk_length
        self.topk = topk
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = x

        # Build adjacency matrix
        adj = torch.zeros(N, N, device=x.device)
        src, dst = edge_index
        adj[src, dst] = 1.0

        # Row-normalize
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)

        # Soft TopK anchoring via Jaccard similarity
        topk_mask = torch.zeros_like(adj)
        for i in range(N):
            neighbors = adj[i].nonzero(as_tuple=False).view(-1)
            if neighbors.numel() == 0:
                continue
            sims = torch.zeros_like(neighbors, dtype=torch.float)
            for j_idx, j in enumerate(neighbors):
                sims[j_idx] = (adj[i] * adj[j]).sum() / ((adj[i] + adj[j] - adj[i]*adj[j]).sum() + 1e-6)
            topk_idx = sims.topk(min(self.topk, len(sims)))[1]
            topk_mask[i, neighbors[topk_idx]] = 1.0
        adj = adj * topk_mask
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)

        # Walk aggregation
        for _ in range(self.walk_length):
            h = adj @ h

        # Linear projection
        h = self.lin(h)
        return h

# -------------------------------
# Unsupervised contrastive loss (InfoNCE)
# -------------------------------
def contrastive_loss_unsupervised(h1, h2, tau=0.5):
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)

    sim_matrix = h1 @ h2.t()  # cosine similarity
    N = h1.size(0)
    numerator = torch.exp(torch.diag(sim_matrix) / tau)
    denominator = torch.exp(sim_matrix / tau).sum(dim=1)
    loss = -torch.log(numerator / (denominator + 1e-8)).mean()
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
model = TopKRankWalkConv(in_dim=x.size(1), out_dim=64, walk_length=5, topk=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training loop
# -------------------------------
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Augment features with small noise for contrastive learning
    x1 = x + 0.01 * torch.randn_like(x)
    x2 = x + 0.01 * torch.randn_like(x)

    emb1 = model(x1, edge_index_sub)
    emb2 = model(x2, edge_index_sub)

    loss = contrastive_loss_unsupervised(emb1, emb2)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Evaluation
# -------------------------------
with torch.no_grad():
    embeddings = model(x, edge_index_sub).numpy()

# KMeans clustering
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
y_pred_kmeans = kmeans.labels_

# Ward clustering
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward').fit(embeddings)
y_pred_ward = ward.labels_

print("\nContrastive TopK RankWalk -> KMeans ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_kmeans),
    NMI(y.numpy(), y_pred_kmeans)
))

print("Contrastive TopK RankWalk -> Ward ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_ward),
    NMI(y.numpy(), y_pred_ward)
))