# test12.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

# -------------------------------
# Learnable similarity for TopKGraphs
# -------------------------------
class LearnableSimilarity(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures positive
        )

    def forward(self, x_s, x_v):
        # x_s: (F,), x_v: (N,F)
        s_expand = x_s.unsqueeze(0).expand_as(x_v)
        inp = torch.cat([s_expand, x_v], dim=1)
        return self.mlp(inp).squeeze(-1)  # (N,)

# -------------------------------
# TopKGraphs Walk with learnable similarity
# -------------------------------
def topk_walks(x, edge_index, sim_func, walk_length=5, num_walks=5):
    N = x.size(0)
    adj_list = [[] for _ in range(N)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        adj_list[u].append(v)

    # Borda score matrix
    Borda = torch.zeros(N, N, device=x.device)

    for s in range(N):
        first_visits = torch.full((N,), fill_value=walk_length + 1, device=x.device)
        for _ in range(num_walks):
            u = s
            visited = set([u])
            first_visits[u] = min(first_visits[u], 1)
            for t in range(1, walk_length + 1):
                neighbors = adj_list[u]
                if len(neighbors) == 0:
                    break
                probs = sim_func(x[s], x[torch.tensor(neighbors, device=x.device)])
                probs = probs / (probs.sum() + 1e-6)
                idx = torch.multinomial(probs, 1).item()
                v = neighbors[idx]
                if v not in visited:
                    first_visits[v] = min(first_visits[v], t + 1)
                    visited.add(v)
                u = v
        Borda[s] = first_visits.float()
    # Normalize Borda
    Borda = Borda / Borda.max(dim=1, keepdim=True)[0]
    return Borda

# -------------------------------
# Contrastive Loss
# -------------------------------
def contrastive_loss(emb, tau=0.5):
    sim = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    pos_mask = torch.eye(emb.size(0), device=emb.device)
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

# Downsample for speed
num_nodes = 100
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
# Model: Learnable similarity + projection
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
y = y.to(device)
edge_index_sub = edge_index_sub.to(device)

sim_func = LearnableSimilarity(x.size(1), hidden_dim=32).to(device)
proj = nn.Linear(x.size(0), 32).to(device)  # project Borda matrix to embedding

optimizer = torch.optim.Adam(list(sim_func.parameters()) + list(proj.parameters()), lr=0.01)

# -------------------------------
# Training loop
# -------------------------------
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    Borda = topk_walks(x, edge_index_sub, sim_func, walk_length=5, num_walks=5)
    embeddings = proj(Borda)
    loss = contrastive_loss(embeddings)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Evaluation
# -------------------------------
with torch.no_grad():
    Borda = topk_walks(x, edge_index_sub, sim_func, walk_length=5, num_walks=5)
    embeddings = proj(Borda).cpu().numpy()

kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y))).fit(embeddings)

print("\nContrastive Learnable TopK RankWalk:")
print("-> KMeans ARI: {:.3f}, NMI: {:.3f}".format(ARI(y.cpu().numpy(), kmeans.labels_),
                                                  NMI(y.cpu().numpy(), kmeans.labels_)))
print("-> Ward   ARI: {:.3f}, NMI: {:.3f}".format(ARI(y.cpu().numpy(), ward.labels_),
                                                  NMI(y.cpu().numpy(), ward.labels_)))