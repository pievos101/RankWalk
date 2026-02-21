# learnable_topkgraphs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

# -------------------------------
# Learnable TopKGraphs Module
# -------------------------------
class LearnableTopKGraphs(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, walk_length=5):
        super().__init__()
        self.walk_length = walk_length
        # Learnable "Jaccard" similarity: MLP taking concatenated start & neighbor features
        self.sim_mlp = nn.Sequential(
            nn.Linear(2*in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Optional projection after aggregation
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = x

        # Build adjacency list
        adj_list = [[] for _ in range(N)]
        src, dst = edge_index
        for u, v in zip(src.tolist(), dst.tolist()):
            adj_list[u].append(v)

        # Precompute node features for all nodes
        device = x.device
        H = torch.zeros(N, self.proj.out_features, device=device)

        # Start-node anchored walks
        for s in range(N):
            h_s = x[s].unsqueeze(0)  # (1, F)
            scores = torch.zeros(N, device=device) + 1e-6  # small epsilon for unvisited
            visited = torch.zeros(N, device=device).bool()
            current_nodes = [s]
            for t in range(self.walk_length):
                next_nodes = []
                for u in current_nodes:
                    neighbors = adj_list[u]
                    if len(neighbors) == 0:
                        continue
                    # Compute learnable similarity with neighbors
                    feats = torch.cat([h_s.repeat(len(neighbors),1),
                                       x[neighbors]], dim=1)
                    sim = self.sim_mlp(feats).squeeze()  # shape: (#neighbors,)
                    prob = F.softmax(sim, dim=0)
                    # Sample one neighbor per current node
                    choice = torch.multinomial(prob, 1).item()
                    v = neighbors[choice]
                    if not visited[v]:
                        scores[v] = t + 1  # first-visit order
                        visited[v] = True
                    next_nodes.append(v)
                if len(next_nodes) == 0:
                    break
                current_nodes = next_nodes
            # Assign worst rank to unvisited
            scores[~visited] = self.walk_length + 1
            # Aggregate (soft Borda-like)
            H[s] = self.proj(scores.unsqueeze(-1))[:,0]  # linear projection

        return H  # (N, hidden_dim)

# -------------------------------
# Contrastive loss
# -------------------------------
def contrastive_loss(h, tau=0.5):
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
    pos_mask = torch.eye(h.size(0), device=h.device)
    neg_mask = 1 - pos_mask
    loss = (pos_mask * (1 - sim) + neg_mask * F.relu(sim - tau)).mean()
    return loss

# -------------------------------
# Load dataset
# -------------------------------
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
x, y = data.x, data.y
edge_index = data.edge_index

# Downsample for quick testing
num_nodes = min(1000, x.size(0))
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
y = y.to(device)
edge_index_sub = edge_index_sub.to(device)

model = LearnableTopKGraphs(in_dim=x.size(1), hidden_dim=64, walk_length=5).to(device)
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
# Evaluation
# -------------------------------
with torch.no_grad():
    embeddings = model(x, edge_index_sub).cpu().numpy()

# KMeans
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
y_pred_k = kmeans.labels_

# Ward clustering
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward').fit(embeddings)
y_pred_w = ward.labels_

print("\nContrastive Learnable TopK RankWalk:")
print(f"-> KMeans ARI: {ARI(y.cpu().numpy(), y_pred_k):.3f}, NMI: {NMI(y.cpu().numpy(), y_pred_k):.3f}")
print(f"-> Ward   ARI: {ARI(y.cpu().numpy(), y_pred_w):.3f}, NMI: {NMI(y.cpu().numpy(), y_pred_w):.3f}")