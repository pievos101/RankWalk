import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

# -------------------------------
# Differentiable TopKGraphs / Soft Walk
# -------------------------------
class SoftTopKGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, walk_steps=20):
        super().__init__()
        self.walk_steps = walk_steps
        # Learnable similarity function for neighbor pairs
        self.sim_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 1, hidden_dim),  # concat(h_u, h_v, J_uv)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = x.clone()

        # Build adjacency list
        adj = [[] for _ in range(N)]
        src, dst = edge_index
        for u, v in zip(src.tolist(), dst.tolist()):
            adj[u].append(v)
            adj[v].append(u)  # undirected

        # Precompute Jaccard similarity for each edge
        neighbors = [set(adj[i]) for i in range(N)]
        edge_jaccard = []
        for u, v in zip(src.tolist(), dst.tolist()):
            union = len(neighbors[u] | neighbors[v])
            inter = len(neighbors[u] & neighbors[v])
            edge_jaccard.append(inter / union if union > 0 else 0.0)
        edge_jaccard = torch.tensor(edge_jaccard, device=x.device, dtype=torch.float)

        # Walk / propagation
        for step in range(self.walk_steps):
            h_new = torch.zeros_like(h)
            for u in range(N):
                if len(adj[u]) == 0:
                    h_new[u] = h[u]
                    continue
                neighbors_u = adj[u]
                feats = torch.stack([torch.cat([h[u], h[v], torch.tensor([edge_jaccard[i]], device=h.device)])
                                     for i, v in enumerate(neighbors_u)], dim=0)
                attn = F.softmax(self.sim_mlp(feats).squeeze(), dim=0)
                agg = sum(attn[i] * h[neighbors_u[i]] for i in range(len(neighbors_u)))
                h_new[u] = agg
            h = h_new

        # Project to final embedding
        embeddings = self.proj(h)
        return embeddings

# -------------------------------
# Contrastive loss
# -------------------------------
def contrastive_loss(embeddings, margin=0.5):
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    N = embeddings.size(0)
    # simple negative-positive sampling: treat all off-diagonal as negative
    pos_mask = torch.eye(N, device=embeddings.device)
    neg_mask = 1 - pos_mask
    loss = (pos_mask * (1 - sim_matrix) + neg_mask * F.relu(sim_matrix - margin)).mean()
    return loss

# -------------------------------
# Load dataset
# -------------------------------
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
x, y = data.x, data.y
edge_index = data.edge_index

# Downsample
num_nodes = 200
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
# Model & optimizer
# -------------------------------
model = SoftTopKGraph(in_dim=x.size(1), hidden_dim=32, walk_steps=30)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training loop
# -------------------------------
epochs = 50
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

kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward').fit(embeddings)

print("\nContrastive Soft TopK RankWalk:")
print(f"-> KMeans ARI: {ARI(y.numpy(), kmeans.labels_):.3f}, NMI: {NMI(y.numpy(), kmeans.labels_):.3f}")
print(f"-> Ward   ARI: {ARI(y.numpy(), ward.labels_):.3f}, NMI: {NMI(y.numpy(), ward.labels_):.3f}")