# test11.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

# -------------------------------
# Utility: compute Jaccard similarity wrt start node
# -------------------------------
def jaccard_similarity(edge_index, num_nodes, start_node):
    N = num_nodes
    adj = [set() for _ in range(N)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].add(v)
        adj[v].add(u)  # undirected
    start_neighbors = adj[start_node]
    J = torch.zeros(N)
    for v in range(N):
        union = start_neighbors | adj[v]
        inter = start_neighbors & adj[v]
        J[v] = len(inter) / (len(union) + 1e-6)
    return J

# -------------------------------
# TopKGraphs with learnable projection
# -------------------------------
class LearnableTopKGraphs(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32, num_walks=5, walk_length=5, eps=1e-3):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.eps = eps
        # Projection from node-centric Borda scores to embedding
        self.proj = nn.Linear(num_nodes, hidden_dim)
        
    def forward(self, x, edge_index):
        N = self.num_nodes
        device = x.device
        Borda = torch.zeros(N, N, device=device)
        
        src, dst = edge_index
        adj = [[] for _ in range(N)]
        for u, v in zip(src.tolist(), dst.tolist()):
            adj[u].append(v)
            adj[v].append(u)
        
        for s in range(N):
            rank_accum = torch.zeros(N, device=device)
            for _ in range(self.num_walks):
                J = jaccard_similarity(edge_index, N, s).to(device)
                visited = set()
                order = []
                current = s
                for _ in range(self.walk_length):
                    neighbors = adj[current]
                    if len(neighbors) == 0:
                        break
                    probs = torch.tensor([J[n] + self.eps for n in neighbors], device=device)
                    probs = probs / probs.sum()
                    idx = torch.multinomial(probs, 1).item()
                    current = neighbors[idx]
                    if current not in visited:
                        visited.add(current)
                        order.append(current)
                # Extend ranking
                tau = torch.full((N,), len(order)+1, device=device)
                for rank, node in enumerate(order):
                    tau[node] = rank + 1
                rank_accum += tau
            # Average over walks
            Borda[s] = rank_accum / self.num_walks
        
        # Normalize Borda
        Borda = Borda / (Borda.max(dim=1, keepdim=True)[0] + 1e-6)
        
        # Learnable projection
        embeddings = self.proj(Borda)  # (N, hidden_dim)
        return embeddings

# -------------------------------
# Contrastive loss
# -------------------------------
def contrastive_loss(h, margin=0.5):
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)  # (N,N)
    # Self-supervised: positive = diagonal (each node vs itself)
    pos_mask = torch.eye(h.size(0), device=h.device)
    neg_mask = 1 - pos_mask
    loss = (pos_mask * (1 - sim) + neg_mask * F.relu(sim - margin)).mean()
    return loss

# -------------------------------
# Load Cora
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
# Model and optimizer
# -------------------------------
hidden_dim = 32
model = LearnableTopKGraphs(num_nodes=num_nodes, hidden_dim=hidden_dim, num_walks=5, walk_length=5)
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
    embeddings = embeddings.numpy()

# KMeans
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
y_pred_k = kmeans.labels_

# Ward
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward').fit(embeddings)
y_pred_w = ward.labels_

print("\nContrastive Learnable TopK RankWalk:")
print("-> KMeans ARI: {:.3f}, NMI: {:.3f}".format(ARI(y.numpy(), y_pred_k), NMI(y.numpy(), y_pred_k)))
print("-> Ward   ARI: {:.3f}, NMI: {:.3f}".format(ARI(y.numpy(), y_pred_w), NMI(y.numpy(), y_pred_w)))