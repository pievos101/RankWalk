import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_undirected
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from node2vec import Node2Vec
import networkx as nx
import random
from collections import defaultdict

# --------------------------
# Load graph
# --------------------------
dataset = KarateClub()
data = dataset[0]
x = data.x
edge_index = to_undirected(data.edge_index)
labels = data.y
num_nodes = x.size(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return F.normalize(h, dim=1)

model = GNN(x.size(1)).to(device)

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
            if union:
                J[u, v] = len(Nu & Nv) / len(union)
    return J

J = jaccard_similarity(edge_index, num_nodes)

# --------------------------
# Borda Top-K pairs
# --------------------------
def borda_topk_pairs(J, edge_index, walk_length=50, num_walks=100, top_k=3, eps=1e-6):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)

    pos_pairs = []

    for start in range(num_nodes):
        rank_sum = defaultdict(float)
        visit_count = defaultdict(int)

        for _ in range(num_walks):
            visited = {}
            cur = start
            visited[cur] = 1

            for step in range(1, walk_length + 1):
                if not neighbors[cur]:
                    break
                probs = torch.tensor(
                    [J[start, n].item() + eps for n in neighbors[cur]],
                    device=device
                )
                probs /= probs.sum()
                nxt = random.choices(neighbors[cur], weights=probs.cpu().tolist(), k=1)[0]
                if nxt not in visited:
                    visited[nxt] = step + 1
                cur = nxt

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
# Contrastive loss
# --------------------------
def contrastive_loss_borda(emb, pos_pairs, neg_multiplier=2):
    device = emb.device

    pos_i = torch.tensor([p[0] for p in pos_pairs], device=device)
    pos_j = torch.tensor([p[1] for p in pos_pairs], device=device)

    pos_sim = (emb[pos_i] * emb[pos_j]).sum(dim=1)
    pos_loss = (1 - pos_sim).mean()

    neg_i = torch.randint(0, num_nodes, (len(pos_i) * neg_multiplier,), device=device)
    neg_j = torch.randint(0, num_nodes, (len(pos_j) * neg_multiplier,), device=device)

    neg_sim = (emb[neg_i] * emb[neg_j]).sum(dim=1)
    neg_loss = F.relu(neg_sim).mean()

    return pos_loss + neg_loss

# --------------------------
# Training
# --------------------------
def train(model, x, edge_index, epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_pairs = borda_topk_pairs(J, edge_index)

    best_loss = float("inf")
    best_emb = None

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index)
        loss = contrastive_loss_borda(emb, pos_pairs)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()

    return best_emb

# --------------------------
# Evaluation
# --------------------------
def evaluate(emb, labels, name):
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    k = len(set(labels))

    km = KMeans(n_clusters=k, n_init=20).fit(emb)
    ward = AgglomerativeClustering(n_clusters=k).fit(emb)

    print(f"\n{name}")
    print(f"KMeans ARI: {adjusted_rand_score(labels, km.labels_):.3f}, "
          f"NMI: {normalized_mutual_info_score(labels, km.labels_):.3f}")
    print(f"Ward   ARI: {adjusted_rand_score(labels, ward.labels_):.3f}, "
          f"NMI: {normalized_mutual_info_score(labels, ward.labels_):.3f}")

# --------------------------
# Node2Vec baseline
# --------------------------
def run_node2vec(edge_index, num_nodes, dim=48):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().cpu().numpy())

    node2vec = Node2Vec(
        G,
        dimensions=dim,
        walk_length=20,
        num_walks=200,
        p=1.0,
        q=1.0,
        workers=1,
        #seed=42
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=128)

    emb = torch.zeros(num_nodes, dim)
    for i in range(num_nodes):
        emb[i] = torch.tensor(model.wv[str(i)])

    return emb

# --------------------------
# Run
# --------------------------
emb_gnn = train(model, x, edge_index)
evaluate(emb_gnn, labels, "TopKGraphs-style GNN")

emb_n2v = run_node2vec(edge_index, num_nodes)
evaluate(emb_n2v, labels, "Node2Vec baseline")