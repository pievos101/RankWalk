import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

# -------------------------------
# Learnable TopKGraphs
# -------------------------------
class LearnableTopKGraphs(nn.Module):
    def __init__(self, num_nodes, walk_length=5, K=10, embed_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.K = K
        # Learnable per-node scaling on Jaccard
        self.node_weights = nn.Parameter(torch.ones(num_nodes))
        # Optional linear projection
        self.proj = nn.Linear(num_nodes, embed_dim)

    def forward(self, x, edge_index):
        N = self.num_nodes
        device = x.device

        # Build adjacency
        adj = torch.zeros(N, N, device=device)
        src, dst = edge_index
        adj[src, dst] = 1.0

        # Precompute neighborhoods
        neighbors = [set((dst[src == i]).tolist()) for i in range(N)]

        # Compute Jaccard for all start nodes
        J = torch.zeros(N, N, device=device)
        for s in range(N):
            Ns = neighbors[s]
            for v in range(N):
                Nv = neighbors[v]
                union = len(Ns | Nv)
                inter = len(Ns & Nv)
                J[s, v] = inter / union if union > 0 else 0.0
        # Apply learnable node weights
        J = J * self.node_weights.unsqueeze(0)  # scale columns by weight

        # Perform K walks per node and Borda aggregation
        Borda = torch.zeros(N, N, device=device)
        eps = 1e-6
        for s in range(N):
            scores = torch.zeros(N, device=device)
            for k in range(self.K):
                visited = torch.zeros(N, dtype=torch.bool, device=device)
                node = s
                visited[node] = True
                rank = 1
                scores[node] += rank
                for t in range(self.walk_length):
                    nbrs = (adj[node] > 0).nonzero(as_tuple=True)[0]
                    if len(nbrs) == 0:
                        break
                    prob = J[s, nbrs] + eps
                    prob = prob / prob.sum()
                    node = nbrs[torch.multinomial(prob, 1).item()]
                    if not visited[node]:
                        rank += 1
                        visited[node] = True
                        scores[node] += rank
                # penalize unvisited nodes
                scores[~visited] += rank + 1
            Borda[s] = scores / self.K
        # Normalize
        Borda = Borda / Borda.max(dim=1, keepdim=True)[0]
        # Linear projection for embeddings
        embeddings = self.proj(Borda)
        return embeddings

# -------------------------------
# Contrastive loss on embeddings
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LearnableTopKGraphs(num_nodes=num_nodes, walk_length=5, K=5, embed_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
x = x.to(device)
edge_index_sub = edge_index_sub.to(device)

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
embeddings = embeddings.detach().cpu().numpy()
kmeans = KMeans(n_clusters=len(torch.unique(y)), random_state=42).fit(embeddings)
ward = AgglomerativeClustering(n_clusters=len(torch.unique(y))).fit(embeddings)
y_pred_k = kmeans.labels_
y_pred_w = ward.labels_

print("\nContrastive Learnable TopK RankWalk:")
print(f"-> KMeans ARI: {ARI(y.numpy(), y_pred_k):.3f}, NMI: {NMI(y.numpy(), y_pred_k):.3f}")
print(f"-> Ward   ARI: {ARI(y.numpy(), y_pred_w):.3f}, NMI: {NMI(y.numpy(), y_pred_w):.3f}")