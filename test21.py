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
# Load the graph
# --------------------------
dataset = KarateClub()
data = dataset[0]
x = data.x
edge_index = to_undirected(data.edge_index)
labels = data.y

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
# Jaccard Similarity Precomputation
# --------------------------
def jaccard_similarity_matrix(edge_index, num_nodes, eps=1e-6):
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    J = torch.zeros((num_nodes, num_nodes), device=device)
    for i in range(num_nodes):
        for j in range(num_nodes):
            union = neighbors[i] | neighbors[j]
            inter = neighbors[i] & neighbors[j]
            J[i,j] = len(inter) / (len(union) + eps)
    return J

J = jaccard_similarity_matrix(edge_index, x.size(0))

# --------------------------
# Generate co-occurrence pairs from Jaccard-anchored random walks
# --------------------------
def generate_positive_pairs(J, edge_index, num_walks=50, walk_length=5, eps=1e-6):
    num_nodes = J.size(0)
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    pos_pairs = set()
    for start in range(num_nodes):
        for _ in range(num_walks):
            walk = [start]
            current = start
            for _ in range(walk_length):
                nbs = list(neighbors[current])
                if not nbs:
                    break
                probs = torch.tensor([J[start, nb]+eps for nb in nbs], device=device)
                probs = probs / probs.sum()
                next_node = random.choices(nbs, weights=probs.cpu().numpy())[0]
                walk.append(next_node)
                current = next_node
            # Add all co-occurrence pairs in this walk
            for i in range(len(walk)):
                for j in range(i+1, len(walk)):
                    pos_pairs.add((walk[i], walk[j]))
    # Convert to tensors
    pos_i = torch.tensor([i for i,j in pos_pairs], device=device)
    pos_j = torch.tensor([j for i,j in pos_pairs], device=device)
    return pos_i, pos_j

pos_i, pos_j = generate_positive_pairs(J, edge_index, num_walks=20, walk_length=20)

# --------------------------
# Contrastive Loss using co-occurrence pairs
# --------------------------
def cooccurrence_contrastive_loss(emb, pos_i, pos_j, neg_multiplier=2):
    # Positive pairs
    pos_sim = (emb[pos_i] * emb[pos_j]).sum(dim=1)
    pos_loss = (1 - pos_sim).mean()
    
    # Negative pairs: sample randomly
    num_nodes = emb.size(0)
    num_neg = len(pos_i) * neg_multiplier
    neg_i = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_j = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_sim = (emb[neg_i] * emb[neg_j]).sum(dim=1)
    neg_loss = F.relu(neg_sim).mean()
    
    return pos_loss + neg_loss

# --------------------------
# Training
# --------------------------
def train(model, x, edge_index, pos_i, pos_j, epochs=800, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        emb = model(x, edge_index)
        loss = cooccurrence_contrastive_loss(emb, pos_i, pos_j)
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
    
    print("Self-Supervised Embeddings:")
    print(f"-> KMeans ARI: {adjusted_rand_score(labels, km.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, km.labels_):.3f}")
    print(f"-> Ward   ARI: {adjusted_rand_score(labels, ward.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, ward.labels_):.3f}")

# --------------------------
# Run training
# --------------------------
embeddings = train(model, x, edge_index, pos_i, pos_j, epochs=800, lr=0.01)
evaluate(embeddings, labels)