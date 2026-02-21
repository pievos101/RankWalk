import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_undirected
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random

# -----------------------------
# Load graph
# -----------------------------
dataset = KarateClub()
data = dataset[0]
x = data.x
edge_index = to_undirected(data.edge_index)
labels = data.y
num_nodes = x.size(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, edge_index, labels = x.to(device), edge_index.to(device), labels.to(device)

# -----------------------------
# Model
# -----------------------------
class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=48):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        return F.normalize(h2, dim=1)  # normalized embeddings

model = GNN(x.size(1), hidden_dim=48).to(device)

# -----------------------------
# Jaccard similarity matrix
# -----------------------------
def jaccard_matrix(edge_index, num_nodes):
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    J = torch.zeros((num_nodes, num_nodes), device=device)
    for i in range(num_nodes):
        for j in range(num_nodes):
            union = neighbors[i] | neighbors[j]
            inter = neighbors[i] & neighbors[j]
            if union:
                J[i, j] = len(inter) / len(union)
    return J

J = jaccard_matrix(edge_index, num_nodes)

# -----------------------------
# Generate positive pairs from Jaccard random walks
# -----------------------------
def jaccard_walk_pairs(J, walk_length=4, num_walks=10, eps=1e-6, decay=0.7):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
    
    src_list, dst_list, weight_list = [], [], []

    for start in range(num_nodes):
        for _ in range(num_walks):
            walk = [start]
            cur = start
            for t in range(walk_length):
                nbrs = neighbors[cur]
                if not nbrs:
                    break
                probs = torch.tensor([J[start, n].item() + eps for n in nbrs], device=device)
                probs = probs / probs.sum()
                next_node = random.choices(nbrs, weights=probs.cpu().tolist(), k=1)[0]
                walk.append(next_node)
                cur = next_node
            # Co-occurrence pairs with temporal decay
            for i in range(len(walk)):
                for j in range(i + 1, len(walk)):
                    t_dist = j - i
                    w = torch.exp(torch.tensor(-decay * t_dist, device=device))
                    src_list.append(walk[i])
                    dst_list.append(walk[j])
                    weight_list.append(w)

    src = torch.tensor(src_list, device=device)
    dst = torch.tensor(dst_list, device=device)
    weights = torch.stack(weight_list)
    return src, dst, weights

# -----------------------------
# Combined contrastive loss
# -----------------------------
def contrastive_jaccard_loss(emb, src, dst, weights, neg_multiplier=2):
    # Positive pairs
    pos_sim = (emb[src] * emb[dst]).sum(dim=1)
    pos_loss = ((1 - pos_sim) * weights).mean()

    # Negative sampling: pick random nodes that do not co-occur
    num_nodes = emb.size(0)
    neg_i = torch.randint(0, num_nodes, (len(src) * neg_multiplier,), device=device)
    neg_j = torch.randint(0, num_nodes, (len(dst) * neg_multiplier,), device=device)
    neg_sim = (emb[neg_i] * emb[neg_j]).sum(dim=1)
    neg_loss = F.relu(neg_sim).mean()

    return pos_loss + neg_loss

# -----------------------------
# Training
# -----------------------------
def train(model, x, edge_index, epochs=500, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    # Precompute positive pairs once
    src, dst, weights = jaccard_walk_pairs(J, walk_length=7, num_walks=10)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        emb = model(x, edge_index)
        loss = contrastive_jaccard_loss(emb, src, dst, weights)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()
            
    return best_emb

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(emb, labels):
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    n_clusters = len(set(labels))
    
    km = KMeans(n_clusters=n_clusters, n_init=10).fit(emb)
    ward = AgglomerativeClustering(n_clusters=n_clusters).fit(emb)
    
    print("Self-Supervised Embeddings:")
    print(f"-> KMeans ARI: {adjusted_rand_score(labels, km.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, km.labels_):.3f}")
    print(f"-> Ward   ARI: {adjusted_rand_score(labels, ward.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, ward.labels_):.3f}")

# -----------------------------
# Run
# -----------------------------
embeddings = train(model, x, edge_index, epochs=800, lr=0.01)
evaluate(embeddings, labels)