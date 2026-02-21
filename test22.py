import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_undirected
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random

# --------------------------
# Load graph
# --------------------------
dataset = KarateClub()
data = dataset[0]
x = data.x
edge_index = to_undirected(data.edge_index)
labels = data.y
num_nodes = x.size(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, edge_index, labels = x.to(device), edge_index.to(device), labels.to(device)

# --------------------------
# GNN Model
# --------------------------
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

# --------------------------
# Jaccard Similarity
# --------------------------
def jaccard_similarity(edge_index, num_nodes):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    J = torch.zeros(num_nodes, num_nodes, device=device)
    for u in range(num_nodes):
        Nu = set(neighbors[u])
        for v in range(num_nodes):
            Nv = set(neighbors[v])
            union = Nu | Nv
            inter = Nu & Nv
            if len(union) > 0:
                J[u, v] = len(inter) / len(union)
    return J

# Precompute Jaccard similarity matrix
J = jaccard_similarity(edge_index, num_nodes)

# --------------------------
# Random walk co-occurrences
# --------------------------
def jaccard_walk_pairs(J, edge_index, walk_length=4, num_walks=10, eps=1e-6):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
    
    pos_pairs = []
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
            # Add all co-occurrence pairs in the walk with temporal weighting
            for i in range(len(walk)):
                for j in range(i+1, len(walk)):
                    t_dist = j - i
                    weight = torch.exp(torch.tensor(-0.7 * t_dist, device=device))
                    pos_pairs.append((walk[i], walk[j], weight))
    return pos_pairs

# --------------------------
# Contrastive Loss using walk co-occurrences
# --------------------------
def contrastive_loss_walks(emb, pos_pairs, neg_multiplier=2):
    
    device = emb.device
    pos_i = torch.tensor([p[0] for p in pos_pairs], device=device)
    pos_j = torch.tensor([p[1] for p in pos_pairs], device=device)
    pos_w = torch.tensor([p[2] for p in pos_pairs], device=device)

    pos_sim = (emb[pos_i] * emb[pos_j]).sum(dim=1)
    pos_loss = ((1 - pos_sim) * pos_w).mean()

    # Negative sampling: pick random nodes
    num_nodes = emb.size(0)
    neg_i = torch.randint(0, num_nodes, (len(pos_i)*neg_multiplier,), device=device)
    neg_j = torch.randint(0, num_nodes, (len(pos_j)*neg_multiplier,), device=device)
    neg_sim = (emb[neg_i] * emb[neg_j]).sum(dim=1)
    neg_loss = F.relu(neg_sim).mean()

    return pos_loss + neg_loss

# --------------------------
# Training
# --------------------------
def train(model, x, edge_index, epochs=300, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    
    # Precompute positive pairs from Jaccard walks
    pos_pairs = jaccard_walk_pairs(J, edge_index, walk_length=5, num_walks=50)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        emb = model(x, edge_index)
        loss = contrastive_loss_walks(emb, pos_pairs)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()
            
    return best_emb

# --------------------------
# Evaluation
# --------------------------
def evaluate(emb, labels):
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    n_clusters = len(set(labels))
    
    km = KMeans(n_clusters=n_clusters, n_init=10).fit(emb)
    ward = AgglomerativeClustering(n_clusters=n_clusters).fit(emb)
    
    print("TopKGraphs Self-Supervised Embeddings:")
    print(f"-> KMeans ARI: {adjusted_rand_score(labels, km.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, km.labels_):.3f}")
    print(f"-> Ward   ARI: {adjusted_rand_score(labels, ward.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, ward.labels_):.3f}")

# --------------------------
# Run
# --------------------------
embeddings = train(model, x, edge_index, epochs=300, lr=0.001)
evaluate(embeddings, labels)