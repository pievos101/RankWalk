import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from torch_geometric.datasets import Planetoid
import numpy as np

# -------------------------------
# Learnable Top-K + Anchored RankWalk
# -------------------------------
class LearnableTopKRankWalk(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, walk_length=5, top_k=10):
        super().__init__()
        self.walk_length = walk_length
        self.top_k = top_k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = self.mlp(x)  # node embeddings for adjacency similarity

        # Compute soft adjacency (cosine similarity)
        sim_matrix = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix * (torch.eye(N, device=x.device) == 0)  # zero diagonal

        # Top-K per row
        topk_mask = torch.zeros_like(sim_matrix)
        for i in range(N):
            if sim_matrix[i].sum() > 0:
                topk_idx = torch.topk(sim_matrix[i], min(self.top_k, N))[1]
                topk_mask[i, topk_idx] = 1.0
        adj = sim_matrix * topk_mask

        # Row-normalize for walks
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)

        # Anchored first-visit walks
        tau = torch.zeros(N, N, device=x.device)
        for anchor in range(N):
            q = torch.zeros(N, device=x.device)
            q[anchor] = 1.0
            visited = torch.zeros(N, device=x.device)
            for _ in range(self.walk_length):
                visited += q
                q = adj @ q
            tau[anchor] = visited

        # Optional linear projection for downstream tasks
        out = self.lin_out(tau)
        return out, tau

# -------------------------------
# Contrastive Loss
# -------------------------------
def contrastive_loss(h, y, margin=0.5):
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
    y = y.view(-1)
    pos_mask = (y.unsqueeze(1) == y.unsqueeze(0)).float()
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

# Downsample for faster testing
num_nodes = 500
idx = np.random.choice(x.size(0), num_nodes, replace=False)
x = x[idx]
y = y[idx]

# Map edges to downsampled nodes
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
model = LearnableTopKRankWalk(in_dim=x.size(1), hidden_dim=64, walk_length=50, top_k=30)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training loop
# -------------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    h, tau = model(x, edge_index_sub)
    loss = contrastive_loss(h, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Clustering evaluation
# -------------------------------
with torch.no_grad():
    embeddings, tau = model(x, edge_index_sub)
    embeddings = embeddings.numpy()
    tau = tau.numpy()

# KMeans
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
y_pred_k = kmeans.labels_

# Ward (AgglomerativeClustering)
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