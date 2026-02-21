import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.nn import MessagePassing
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
    probs = [[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    labels = []
    for idx, size in enumerate(sizes):
        labels.extend([idx] * size)
    return G, torch.tensor(labels, dtype=torch.long)

# --------------------------
# Sparse Jaccard edge weights
# --------------------------
def compute_edge_jaccard(edge_index, num_nodes):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
        neighbors[v].append(u)
    edge_weight = []
    for u, v in edge_index.t().tolist():
        Nu, Nv = set(neighbors[u]), set(neighbors[v])
        union = Nu | Nv
        w = len(Nu & Nv) / len(union) if union else 0.0
        edge_weight.append(w)
    return torch.tensor(edge_weight, dtype=torch.float)

# --------------------------
# GNN with (optionally learnable) edge weights
# --------------------------
class JaccardGNN(MessagePassing):
    def __init__(self, in_dim, hidden_dim=48, out_dim=48, learn_edge=False):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.learn_edge = learn_edge
        if learn_edge:
            self.edge_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.lin1(x))
        if self.learn_edge and edge_weight is not None:
            edge_weight = self.edge_mlp(edge_weight.unsqueeze(1)).squeeze()
        h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
        h = self.lin2(h)
        return F.normalize(h, dim=1)

    def message(self, x_j, edge_weight=None):
        if edge_weight is not None:
            return x_j * edge_weight.unsqueeze(1)
        return x_j

# --------------------------
# Vectorized TopKGraphs-style positive sampling
# --------------------------
def sample_pos_pairs(J, edge_index, num_nodes, walk_length=10, num_walks=10, top_k=3, eps=1e-6):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)
    pos_pairs = []
    for start in range(num_nodes):
        rank_sum = defaultdict(float)
        visit_count = defaultdict(int)
        for _ in range(num_walks):
            cur = start
            visited = {start: 1}
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
        topk = sorted(borda, key=borda.get)[:top_k]
        for i in range(len(topk)):
            for j in range(i + 1, len(topk)):
                pos_pairs.append((topk[i], topk[j]))
    return pos_pairs

# --------------------------
# Efficient batch-wise InfoNCE loss
# --------------------------
def contrastive_loss(emb, pos_pairs, temperature=0.2, neg_multiplier=5):
    device = emb.device
    num_nodes = emb.size(0)
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=device)
    
    pos_i = torch.tensor([p[0] for p in pos_pairs], device=device)
    pos_j = torch.tensor([p[1] for p in pos_pairs], device=device)

    pos_sim = (emb[pos_i] * emb[pos_j]).sum(dim=1) / temperature
    pos_exp = torch.exp(pos_sim)

    # batch-wise negatives
    neg_exp_sum = torch.zeros_like(pos_exp)
    for idx, pi in enumerate(pos_i):
        neg_idx = torch.randint(0, num_nodes, (neg_multiplier,), device=device)
        neg_sim = (emb[pi] * emb[neg_idx]).sum(dim=1) / temperature
        neg_exp_sum[idx] = torch.exp(neg_sim).sum()
    
    loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum)).mean()
    return loss

# --------------------------
# Train GNN end-to-end
# --------------------------
def train_gnn(x, edge_index, edge_weight, J, epochs=500, lr=1e-3, walk_length=10, num_walks=10, top_k=5):
    model = JaccardGNN(x.size(1), out_dim=48, learn_edge=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_emb = None
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index, edge_weight=edge_weight)
        pos_pairs = sample_pos_pairs(J, edge_index, x.size(0), walk_length, num_walks, top_k)
        loss = contrastive_loss(emb, pos_pairs)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | InfoNCE Loss: {loss.item():.4f}")
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
# Run benchmark
# --------------------------
if __name__ == "__main__":
    # SBM graph
    G_nx, labels = generate_sbm_graph(n_communities=3, size_per_comm=20, p_in=0.3, p_out=0.05)
    data = from_networkx(G_nx)
    edge_index = to_undirected(data.edge_index)

    # Node features
    #x = torch.randn(G_nx.number_of_nodes(), 20)
    x = torch.eye(G_nx.number_of_nodes())
    
    # Sparse Jaccard weights
    edge_weight = compute_edge_jaccard(edge_index, G_nx.number_of_nodes())
    J = torch.zeros(G_nx.number_of_nodes(), G_nx.number_of_nodes())
    for idx, (u, v) in enumerate(edge_index.t().tolist()):
        J[u, v] = edge_weight[idx]
        J[v, u] = edge_weight[idx]

    # Train GNN end-to-end
    emb_gnn = train_gnn(x, edge_index, edge_weight, J, epochs=500, 
                            walk_length=10, num_walks=10, top_k=10)
    evaluate(emb_gnn, labels, "TopKGraphs GNN (contrastive)")

    # Node2Vec baseline
    emb_n2v = run_node2vec(G_nx)
    evaluate(emb_n2v, labels, "Node2Vec")