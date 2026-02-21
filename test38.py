import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.nn import MessagePassing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import networkx as nx
from collections import defaultdict
from node2vec import Node2Vec
import random

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
# Fast sparse Jaccard similarity
# --------------------------
def compute_jaccard_fast(edge_index, num_nodes, device='cpu'):
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    row, col, data = [], [], []
    for u in range(num_nodes):
        Nu = neighbors[u]
        candidates = set()
        for v in Nu:
            candidates.update(neighbors[v])
        candidates.discard(u)
        for v in candidates:
            Nv = neighbors[v]
            union = Nu | Nv
            if union:
                row.append(u)
                col.append(v)
                data.append(len(Nu & Nv) / len(union))
    
    if len(row) == 0:
        return torch.zeros((num_nodes, num_nodes), device=device)
    
    J = torch.sparse_coo_tensor([row, col], data, (num_nodes, num_nodes), device=device).to_dense()
    return J

# --------------------------
# Vanilla GNN
# --------------------------
class VanillaGNN(MessagePassing):
    def __init__(self, in_dim, hidden_dim=48, out_dim=48):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.lin1(x))
        h = self.propagate(edge_index, x=h)
        h = self.lin2(h)
        return F.normalize(h, dim=1)

    def message(self, x_j):
        return x_j

# --------------------------
# Vectorized weighted random walks
# --------------------------
def sample_ranked_pos_pairs_fast(J, edge_index, num_nodes, walk_length=10, num_walks=10, top_k=5, eps=1e-6):
    neighbors_list = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors_list[u].append(v)
    neighbors_tensor = [torch.tensor(neigh, dtype=torch.long) for neigh in neighbors_list]

    pos_pairs = []

    for start in range(num_nodes):
        rank_sum = defaultdict(float)
        visit_count = defaultdict(int)

        for _ in range(num_walks):
            cur = start
            visited = {start: 1}
            for step in range(1, walk_length + 1):
                nbrs = neighbors_tensor[cur]
                if nbrs.numel() == 0:
                    break
                probs = J[start, nbrs].to_dense() + eps
                probs /= probs.sum()
                cur = nbrs[torch.multinomial(probs, 1).item()]
                if cur not in visited:
                    visited[cur] = step + 1

            for v, r in visited.items():
                rank_sum[v] += r
                visit_count[v] += 1

        borda = {v: rank_sum[v] / visit_count[v] for v in visit_count}
        topk = sorted(borda, key=borda.get)[:top_k]
        max_rank = max([borda[v] for v in topk])

        for v in topk:
            weight = 1.0 - (borda[v] / (max_rank + 1e-6))
            if v != start:
                pos_pairs.append((start, v, weight))

    return pos_pairs

# --------------------------
# Fixed weighted InfoNCE loss
# --------------------------
def contrastive_loss_weighted_fixed(emb, pos_pairs, temperature=0.5, neg_multiplier=20):
    device = emb.device
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=device)

    pos_i = torch.tensor([p[0] for p in pos_pairs], device=device)
    pos_j = torch.tensor([p[1] for p in pos_pairs], device=device)
    weights = torch.tensor([max(p[2], 0.05) for p in pos_pairs], device=device)

    pos_sim = (emb[pos_i] * emb[pos_j]).sum(dim=1) / temperature
    pos_exp = torch.exp(pos_sim)

    num_nodes = emb.size(0)
    neg_idx = torch.randint(0, num_nodes, (len(pos_i), neg_multiplier), device=device)
    neg_emb = emb[neg_idx]
    pi_emb = emb[pos_i].unsqueeze(1)
    neg_sim = (pi_emb * neg_emb).sum(dim=2) / temperature
    neg_exp_sum = torch.exp(neg_sim).sum(dim=1)

    denom = pos_exp + neg_exp_sum + 1e-8
    loss = -(weights * torch.log(pos_exp / denom)).mean()
    return loss

# --------------------------
# Train GNN end-to-end
# --------------------------
def train_gnn(x, edge_index, J, epochs=500, lr=1e-3, walk_length=10, num_walks=10, top_k=5, device='cpu'):
    x, edge_index, J = x.to(device), edge_index.to(device), J.to(device)
    model = VanillaGNN(x.size(1), out_dim=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_emb = None
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index)
        pos_pairs = sample_ranked_pos_pairs_fast(J, edge_index, x.size(0), walk_length, num_walks, top_k)
        loss = contrastive_loss_weighted_fixed(emb, pos_pairs)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | InfoNCE Loss: {loss.item():.4f}")

    return best_emb

# --------------------------
# Node2Vec baseline
# --------------------------
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G_nx, labels = generate_sbm_graph(n_communities=3, size_per_comm=30, p_in=0.2, p_out=0.05)
    data = from_networkx(G_nx)
    edge_index = to_undirected(data.edge_index)

    x = torch.randn(G_nx.number_of_nodes(), 20)

    num_nodes = G_nx.number_of_nodes()
    J = compute_jaccard_fast(edge_index, num_nodes, device=device)

    emb_gnn = train_gnn(x, edge_index, J, epochs=200, walk_length=10, num_walks=10, top_k=10, device=device)
    evaluate(emb_gnn, labels, "TopKGraphs GNN (contrastive)")

    emb_n2v = run_node2vec(G_nx)
    evaluate(emb_n2v, labels, "Node2Vec")