import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.nn import GraphConv
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import networkx as nx
import random
from collections import defaultdict

# --------------------------
# Synthetic SBM graph generator
# --------------------------
def generate_sbm_graph(n_communities=3, size_per_comm=50, p_in=0.8, p_out=0.05, seed=42):
    sizes = [size_per_comm] * n_communities
    probs = [[p_in if i==j else p_out for j in range(n_communities)] for i in range(n_communities)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    labels = []
    for idx, size in enumerate(sizes):
        labels.extend([idx] * size)
    return G, torch.tensor(labels, dtype=torch.long)

# --------------------------
# GNN Model
# --------------------------
class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=48, out_dim=48):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return F.normalize(h, dim=1)

# --------------------------
# Jaccard similarity
# --------------------------
def jaccard_similarity(edge_index, num_nodes):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
        neighbors[v].append(u)
    J = torch.zeros(num_nodes, num_nodes)
    for u in range(num_nodes):
        Nu = set(neighbors[u])
        for v in range(num_nodes):
            Nv = set(neighbors[v])
            union = Nu | Nv
            if union:
                J[u, v] = len(Nu & Nv) / len(union)
    return J

# --------------------------
# Compute TopKGraphs affinity matrix
# --------------------------
def borda_affinity_matrix(J, edge_index, num_nodes, walk_length=50, num_walks=100, eps=1e-6):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
    A = torch.zeros(num_nodes, num_nodes)
    for start in range(num_nodes):
        rank_sum = defaultdict(float)
        visit_count = defaultdict(int)
        for _ in range(num_walks):
            visited = {start: 1}
            cur = start
            for step in range(1, walk_length + 1):
                nbrs = neighbors[cur]
                if not nbrs:
                    break
                probs = torch.tensor([J[start, n].item() + eps for n in nbrs])
                probs /= probs.sum()
                cur = random.choices(nbrs, weights=probs.tolist(), k=1)[0]
                if cur not in visited:
                    visited[cur] = step + 1
            for v, r in visited.items():
                rank_sum[v] += r
                visit_count[v] += 1
        borda = {v: rank_sum[v] / visit_count[v] for v in visit_count}
        # Convert to affinity: lower Borda rank -> higher affinity
        max_rank = max(borda.values())
        for v, r in borda.items():
            A[start, v] = 1.0 - (r / (max_rank + 1e-6))
    # Symmetrize for downstream tasks
    A = 0.5 * (A + A.T)
    return A

# --------------------------
# GNN training: predict affinity matrix
# --------------------------
def train_gnn_affinity(x, edge_index, A, epochs=500, lr=1e-3):
    model = GNN(x.size(1), out_dim=48)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_emb = None
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index)
        # cosine similarity matrix
        sim = emb @ emb.T
        sim = (sim + 1e-6) / (sim.norm(dim=1, keepdim=True) @ sim.norm(dim=1, keepdim=True).T + 1e-6)
        loss = F.mse_loss(sim, A)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()
        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | MSE Loss: {loss.item():.4f}")
    return best_emb

# --------------------------
# Node2Vec baseline
# --------------------------
from node2vec import Node2Vec
def run_node2vec(G, dim=48):
    node2vec = Node2Vec(G, dimensions=dim, walk_length=20, num_walks=100, p=1, q=1, workers=1, seed=42)
    model = node2vec.fit(window=10, min_count=1, batch_words=128)
    emb = torch.zeros(G.number_of_nodes(), dim)
    for i in range(G.number_of_nodes()):
        emb[i] = torch.tensor(model.wv[str(i)])
    return emb

# --------------------------
# Evaluation
# --------------------------
def evaluate(emb, labels, name="GNN"):
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    k = len(set(labels))
    km = KMeans(n_clusters=k, n_init=10).fit(emb)
    print(f"{name} | KMeans ARI: {adjusted_rand_score(labels, km.labels_):.3f}, "
          f"NMI: {normalized_mutual_info_score(labels, km.labels_):.3f}")

# --------------------------
# Run synthetic SBM benchmark
# --------------------------
if __name__ == "__main__":
    # Generate synthetic SBM
    G_nx, labels = generate_sbm_graph(n_communities=3, size_per_comm=20, p_in=0.5, p_out=0.05)
    data = from_networkx(G_nx)
    edge_index = to_undirected(data.edge_index)

    # Node features: random small vectors
    x = torch.randn(G_nx.number_of_nodes(), 20)

    # Precompute TopKGraphs affinity matrix
    num_nodes = G_nx.number_of_nodes()
    J = jaccard_similarity(edge_index, num_nodes)
    A = borda_affinity_matrix(J, edge_index, num_nodes)

    # Train GNN to predict TopKGraphs affinities
    emb_gnn = train_gnn_affinity(x, edge_index, A, epochs=500)
    evaluate(emb_gnn, labels, "TopKGraphs GNN")

    # Node2Vec baseline
    emb_n2v = run_node2vec(G_nx)
    evaluate(emb_n2v, labels, "Node2Vec")