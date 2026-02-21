import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

# -------------------------------
# Learnable TopK Graph
# -------------------------------
class LearnableTopK(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, topk=5):
        super().__init__()
        self.topk = topk
        # MLP to compute pairwise node similarity (learnable Jaccard)
        self.mlp = nn.Sequential(
            nn.Linear(2*in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        N = x.size(0)
        xi = x.unsqueeze(1).expand(N, N, -1)
        xj = x.unsqueeze(0).expand(N, N, -1)
        sims = self.mlp(torch.cat([xi, xj], dim=-1)).squeeze(-1)  # (N,N)
        sims = (sims + sims.T)/2  # symmetrize
        # Hard top-K neighbors
        topk_idx = sims.topk(k=min(self.topk, N), dim=-1).indices
        edge_list = []
        for i in range(N):
            for j in topk_idx[i]:
                edge_list.append([i, j.item()])
        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # 2 x E
        return edge_index

# -------------------------------
# RankWalk Convolution
# -------------------------------
class TopKRankWalkLearnable(nn.Module):
    def __init__(self, in_dim, out_dim, walk_length=5, topk=5):
        super().__init__()
        self.topk_net = LearnableTopK(in_dim, topk=topk)
        self.lin = nn.Linear(in_dim, out_dim)
        self.walk_length = walk_length

    def forward(self, x):
        N = x.size(0)
        edge_index = self.topk_net(x)
        adj = torch.zeros(N, N, device=x.device)
        src, dst = edge_index
        adj[src, dst] = 1.0
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
        h = x.clone()
        agg = h.clone()
        for _ in range(self.walk_length):
            h = adj @ h
            agg += h
        return self.lin(agg / (self.walk_length + 1))

# -------------------------------
# Contrastive Loss (NT-Xent)
# -------------------------------
def contrastive_loss(h1, h2, temperature=0.5):
    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)
    sim_matrix = h1 @ h2.T  # cosine similarity
    sim_matrix = sim_matrix / temperature
    labels = torch.arange(h1.size(0), device=h1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# -------------------------------
# Load Cora
# -------------------------------
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]
x, y = data.x, data.y

# Downsample nodes for speed
num_nodes = 500
idx = np.random.choice(x.size(0), num_nodes, replace=False)
x = x[idx]
y = y[idx]

# -------------------------------
# Model & optimizer
# -------------------------------
model = TopKRankWalkLearnable(in_dim=x.size(1), out_dim=64, walk_length=5, topk=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# Training
# -------------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    # Two augmented views: feature dropout
    x1 = x * (torch.rand_like(x) > 0.2).float()
    x2 = x * (torch.rand_like(x) > 0.2).float()
    h1 = model(x1)
    h2 = model(x2)
    loss = contrastive_loss(h1, h2)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Evaluation
# -------------------------------
with torch.no_grad():
    embeddings = model(x).numpy()

# KMeans
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
y_pred_k = kmeans.labels_

# Ward
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y)), linkage='ward').fit(embeddings)
y_pred_w = ward.labels_

print("\nContrastive Learnable TopK RankWalk:")
print(f"-> KMeans ARI: {ARI(y.numpy(), y_pred_k):.3f}, NMI: {NMI(y.numpy(), y_pred_k):.3f}")
print(f"-> Ward   ARI: {ARI(y.numpy(), y_pred_w):.3f}, NMI: {NMI(y.numpy(), y_pred_w):.3f}")