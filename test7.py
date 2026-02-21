import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

# -------------------------------
# Utilities
# -------------------------------
def build_topk_jaccard(edge_index, num_nodes, k=5):
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes)
    src, dst = edge_index
    adj[src, dst] = 1.0
    adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
    # Compute Jaccard similarity
    sims = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        neighbors_i = set(torch.where(adj[i] > 0)[0].tolist())
        for j in range(num_nodes):
            neighbors_j = set(torch.where(adj[j] > 0)[0].tolist())
            inter = len(neighbors_i & neighbors_j)
            union = len(neighbors_i | neighbors_j)
            sims[i, j] = inter / union if union > 0 else 0.0
    # TopK per node
    topk_idx = torch.topk(sims, k=min(k, num_nodes), dim=1).indices
    topk_edge_index = []
    for i in range(num_nodes):
        for j in topk_idx[i]:
            topk_edge_index.append([i, j.item()])
    topk_edge_index = torch.tensor(topk_edge_index).T
    return topk_edge_index

def augment_features(x, drop_rate=0.2):
    mask = torch.rand_like(x) > drop_rate
    return x * mask.float()

# -------------------------------
# RankWalk Conv Layer
# -------------------------------
class TopKRankWalk(nn.Module):
    def __init__(self, in_dim, out_dim, walk_length=10):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.walk_length = walk_length

    def forward(self, x, edge_index, num_nodes):
        N = num_nodes
        h = x.clone()
        # Build dense adjacency
        adj = torch.zeros(N, N, device=x.device)
        src, dst = edge_index
        adj[src, dst] = 1.0
        adj = (adj + adj.T) / 2  # symmetrize
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
        # First-visit walk aggregation
        agg = h.clone()
        h_walk = h.clone()
        for _ in range(self.walk_length):
            h_walk = adj @ h_walk
            agg += h_walk
        h_out = self.lin(agg / (self.walk_length + 1))
        return h_out

# -------------------------------
# InfoNCE Contrastive Loss
# -------------------------------
def info_nce_loss(h1, h2, temperature=0.5):
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    sim = h1 @ h2.T  # cosine similarity
    sim = sim / temperature
    labels = torch.arange(h1.size(0), device=h1.device)
    loss = F.cross_entropy(sim, labels)
    return loss

# -------------------------------
# Load Cora
# -------------------------------
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
x, y = data.x, data.y
edge_index = data.edge_index

# Downsample
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
# TopK adjacency
# -------------------------------
topk_edge_index = build_topk_jaccard(edge_index_sub, num_nodes, k=5)

# -------------------------------
# Model & optimizer
# -------------------------------
model = TopKRankWalk(in_dim=x.size(1), out_dim=64, walk_length=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 500

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(epochs):
    optimizer.zero_grad()
    # two augmentations
    x1 = augment_features(x, drop_rate=0.2)
    x2 = augment_features(x, drop_rate=0.2)
    h1 = model(x1, topk_edge_index, num_nodes)
    h2 = model(x2, topk_edge_index, num_nodes)
    loss = info_nce_loss(h1, h2)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Evaluate embeddings
# -------------------------------
with torch.no_grad():
    embeddings = model(x, topk_edge_index, num_nodes).numpy()

# KMeans
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
y_pred_k = kmeans.labels_
print("\nContrastive TopK RankWalk -> KMeans ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_k), NMI(y.numpy(), y_pred_k)
))

# Ward
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y))).fit(embeddings)
y_pred_w = ward.labels_
print("Contrastive TopK RankWalk -> Ward ARI: {:.3f}, NMI: {:.3f}".format(
    ARI(y.numpy(), y_pred_w), NMI(y.numpy(), y_pred_w)
))