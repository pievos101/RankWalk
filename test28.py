import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_undirected
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
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
# GNN
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
# Jaccard similarity
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
# Epoch-wise best positive per node
# --------------------------
def sample_best_positive(J, edge_index, walk_length=10, num_walks=20):
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].append(v)

    pos = {}

    for start in range(num_nodes):
        score = defaultdict(float)

        for _ in range(num_walks):
            cur = start
            for _ in range(walk_length):
                if not neighbors[cur]:
                    break
                probs = torch.tensor(
                    [J[start, n].item() + 1e-6 for n in neighbors[cur]],
                    device=device
                )
                probs /= probs.sum()
                cur = random.choices(
                    neighbors[cur], weights=probs.cpu().tolist(), k=1
                )[0]
                score[cur] += 1

        # exclude self
        score.pop(start, None)
        if score:
            pos[start] = max(score, key=score.get)

    return pos

# --------------------------
# Pure attractive loss (no negatives)
# --------------------------
def attractive_loss(emb, pos_dict):
    i = torch.tensor(list(pos_dict.keys()), device=emb.device)
    j = torch.tensor(list(pos_dict.values()), device=emb.device)

    sim = (emb[i] * emb[j]).sum(dim=1)
    return (1 - sim).mean()

# --------------------------
# Training
# --------------------------
def train(model, x, edge_index, epochs=500, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_emb = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 🔁 resample positives every epoch
        pos_dict = sample_best_positive(J, edge_index)

        emb = model(x, edge_index)
        loss = attractive_loss(emb, pos_dict)

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
def evaluate(emb, labels):
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    n_clusters = len(set(labels))

    km = KMeans(n_clusters=n_clusters, n_init=20).fit(emb)
    ward = AgglomerativeClustering(n_clusters=n_clusters).fit(emb)

    print("\nEvaluation:")
    print(f"KMeans ARI: {adjusted_rand_score(labels, km.labels_):.3f}, "
          f"NMI: {normalized_mutual_info_score(labels, km.labels_):.3f}")
    print(f"Ward   ARI: {adjusted_rand_score(labels, ward.labels_):.3f}, "
          f"NMI: {normalized_mutual_info_score(labels, ward.labels_):.3f}")

# --------------------------
# Run
# --------------------------
embeddings = train(model, x, edge_index)
evaluate(embeddings, labels)