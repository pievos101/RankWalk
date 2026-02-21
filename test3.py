import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from torch_geometric.datasets import Planetoid
import numpy as np

# -------------------------------
# Anchored TopKGraphs + RankWalk Conv
# -------------------------------
class TopKRankWalkConv(nn.Module):
    """
    Learnable RankWalk using Top-K Jaccard graphs + anchored first-visit rank aggregation
    """
    def __init__(self, in_dim, out_dim, walk_length=5, top_k=10):
        super().__init__()
        self.walk_length = walk_length
        self.top_k = top_k
        self.lin = nn.Linear(in_dim, out_dim)
        self.node_emb = nn.Parameter(torch.randn(in_dim))  # learnable bias

    def forward(self, x, edge_index):
        N = x.size(0)
        device = x.device
        h = x

        # -----------------------
        # Compute adjacency with Jaccard weights
        # -----------------------
        adj = torch.zeros(N, N, device=device)
        src, dst = edge_index
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # make undirected

        # row-normalize
        deg = adj.sum(dim=1, keepdim=True) + 1e-6
        adj = adj / deg

        # -----------------------
        # Build Top-K adjacency per node
        # -----------------------
        topk_adj = torch.zeros_like(adj)
        for i in range(N):
            row = adj[i]
            if row.sum() > 0:
                topk_idx = torch.topk(row, min(self.top_k, N))[1]
                topk_adj[i, topk_idx] = row[topk_idx]
        adj = topk_adj

        # -----------------------
        # Anchored walk + first-visit rank aggregation
        # -----------------------
        tau = torch.zeros(N, N, device=device)
        for anchor in range(N):
            q = torch.zeros(N, device=device)
            q[anchor] = 1.0
            visited = torch.zeros(N, device=device)
            for _ in range(self.walk_length):
                visited += q  # accumulate first visits (soft)
                q = adj @ q
            tau[anchor] = visited

        # normalize tau
        tau = tau / (tau.sum(dim=1, keepdim=True) + 1e-6)

        # linear transformation
        out = tau @ h
        return self.lin(out)


# -------------------------------
# Contrastive Loss (unsupervised)
# -------------------------------
def contrastive_loss(h, margin=0.5):
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)  # (N, N)
    pos_mask = (sim > 0.8).float()       # pseudo positives by similarity
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

# Downsample for speed
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
model = TopKRankWalkConv(in_dim=x.size(1), out_dim=64, walk_length=20, top_k=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training loop
# -------------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    embeddings = model(x, edge_index_sub)
    loss = contrastive_loss(embeddings)
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

print("\nContrastive TopK RankWalk -> KMeans ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_k),
    NMI(y.numpy(), y_pred_k)
))

print("Contrastive TopK RankWalk -> Ward ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_w),
    NMI(y.numpy(), y_pred_w)
))