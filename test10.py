# test_learnable_topkgraphs_allinone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from torch_geometric.datasets import Planetoid
import numpy as np

# -------------------------------
# Learnable TopKGraphs Module
# -------------------------------
class LearnableTopKGraphs(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, walk_length=5, num_walks=10):
        super().__init__()
        self.walk_length = walk_length
        self.num_walks = num_walks
        # Learnable similarity function (MLP)
        self.sim_mlp = nn.Sequential(
            nn.Linear(in_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Optional projection for embeddings
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        device = x.device
        adjacency = [[] for _ in range(N)]
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            adjacency[s].append(d)
            adjacency[d].append(s)  # undirected

        # Compute pairwise learned similarity to start nodes
        sims = torch.zeros(N, N, device=device)
        for s in range(N):
            neighbors = adjacency[s]
            if len(neighbors) == 0:
                continue
            x_s = x[s].repeat(len(neighbors), 1)
            x_nb = x[neighbors]
            inp = torch.cat([x_s, x_nb], dim=1)
            sims[s, neighbors] = self.sim_mlp(inp).squeeze()

        sims = F.relu(sims) + 1e-6  # ensure positivity

        # Normalize adjacency row-wise for walks
        A = torch.zeros(N, N, device=device)
        for i in range(N):
            if sims[i].sum() > 0:
                A[i] = sims[i] / sims[i].sum()

        # Perform first-visit TopK random walks
        Borda = torch.zeros(N, N, device=device)
        for s in range(N):
            first_visits = torch.ones(N, device=device) * (self.walk_length+1)
            for _ in range(self.num_walks):
                curr = s
                visited = set([curr])
                for t in range(self.walk_length):
                    probs = A[curr]
                    if probs.sum() == 0:
                        break
                    next_nodes = torch.arange(N, device=device)
                    # Mask visited nodes to avoid revisiting
                    masked_probs = probs.clone()
                    masked_probs[list(visited)] = 0
                    if masked_probs.sum() == 0:
                        break
                    masked_probs /= masked_probs.sum()
                    next_idx = torch.multinomial(masked_probs, 1).item()
                    curr = next_idx
                    if first_visits[curr] > t+1:
                        first_visits[curr] = t+1
                    visited.add(curr)
            Borda[s] = first_visits

        # Project Borda matrix to lower-dimensional embeddings
        embeddings = self.proj(Borda)
        return embeddings

# -------------------------------
# Contrastive Loss
# -------------------------------
def contrastive_loss(h, margin=0.5):
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
    pos_mask = torch.eye(h.size(0), device=h.device)
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

# Downsample for fast testing
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
# Model
# -------------------------------
model = LearnableTopKGraphs(in_dim=x.size(1), hidden_dim=32, walk_length=5, num_walks=10)
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
    embeddings_np = model(x, edge_index_sub).numpy()

# KMeans
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings_np)
y_pred_km = kmeans.labels_

# Ward Agglomerative
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward')
y_pred_ward = ward.fit_predict(embeddings_np)

print("\nContrastive Learnable TopK RankWalk:")
print("-> KMeans ARI: {:.3f}, NMI: {:.3f}".format(ARI(y.numpy(), y_pred_km),
                                                  NMI(y.numpy(), y_pred_km)))
print("-> Ward   ARI: {:.3f}, NMI: {:.3f}".format(ARI(y.numpy(), y_pred_ward),
                                                  NMI(y.numpy(), y_pred_ward)))