import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from node2vec import Node2Vec

from rankwalk.conv import RankWalkConv

# -----------------------------
# Parameters
# -----------------------------
downsample_n = 200
rw_out_dim = 64
walk_length = 10
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

# -----------------------------
# Load Cora
# -----------------------------
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

print(
    f"Dataset Cora: {data.num_nodes} nodes, "
    f"{data.num_features} features, "
    f"{dataset.num_classes} classes"
)

# -----------------------------
# Build NetworkX graph
# -----------------------------
edge_index_np = data.edge_index.cpu().numpy()
G_full = nx.Graph()
G_full.add_edges_from(edge_index_np.T.tolist())

# -----------------------------
# BFS-based downsampling (keeps connectivity)
# -----------------------------
seed_node = np.random.randint(data.num_nodes)
bfs_nodes = list(nx.bfs_tree(G_full, seed_node).nodes())

if len(bfs_nodes) < downsample_n:
    raise RuntimeError("BFS component too small")

perm = np.array(bfs_nodes[:downsample_n])
perm_set = set(perm)

# -----------------------------
# Subgraph edge filtering
# -----------------------------
edges_sub = [
    (u, v) for u, v in G_full.edges()
    if u in perm_set and v in perm_set
]

print(
    f"Downsampled graph: {len(perm)} nodes, {len(edges_sub)} edges"
)

# -----------------------------
# Remap node IDs
# -----------------------------
id_map = {old: new for new, old in enumerate(perm)}

edge_index_mapped = np.array(
    [[id_map[u], id_map[v]] for u, v in edges_sub]
).T

edge_index_torch = torch.tensor(
    np.hstack([edge_index_mapped, edge_index_mapped[::-1]]),
    dtype=torch.long
)

# -----------------------------
# Features & labels
# -----------------------------
X_torch = data.x[perm]
y_true = data.y[perm].cpu().numpy()
num_classes = len(np.unique(y_true))

print(f"Number of true classes: {num_classes}")

# -----------------------------
# Result storage
# -----------------------------
RES_ari = {"RankWalk": [], "Node2Vec": [], "GCN": []}
RES_nmi = {"RankWalk": [], "Node2Vec": [], "GCN": []}

# ============================================================
# RankWalk
# ============================================================
try:
    conv_rw = RankWalkConv(
        in_dim=X_torch.shape[1],
        out_dim=rw_out_dim,
        walk_length=walk_length
    )

    with torch.no_grad():
        emb_rw = conv_rw(X_torch, edge_index_torch).cpu().numpy()

    clustering = AgglomerativeClustering(
        n_clusters=num_classes, linkage="ward"
    )
    y_pred = clustering.fit_predict(emb_rw)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    RES_ari["RankWalk"].append(ari)
    RES_nmi["RankWalk"].append(nmi)

    print(f"RankWalk -> ARI: {ari:.3f}, NMI: {nmi:.3f}")

except Exception as e:
    print(f"RankWalk failed: {e}")

# ============================================================
# Node2Vec (robust to isolated nodes)
# ============================================================
try:
    G_n2v = nx.Graph()
    G_n2v.add_nodes_from(range(len(perm)))
    G_n2v.add_edges_from(edge_index_mapped.T.tolist())

    node2vec = Node2Vec(
        G_n2v,
        dimensions=rw_out_dim,
        walk_length=walk_length,
        num_walks=20,
        workers=1,
        seed=seed
    )

    model = node2vec.fit(window=10, min_count=1)

    emb_n2v = np.zeros((len(perm), rw_out_dim))
    for i in range(len(perm)):
        if str(i) in model.wv:
            emb_n2v[i] = model.wv[str(i)]

    clustering = AgglomerativeClustering(
        n_clusters=num_classes, linkage="ward"
    )
    y_pred = clustering.fit_predict(emb_n2v)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    RES_ari["Node2Vec"].append(ari)
    RES_nmi["Node2Vec"].append(nmi)

    print(f"Node2Vec -> ARI: {ari:.3f}, NMI: {nmi:.3f}")

except Exception as e:
    print(f"Node2Vec failed: {e}")

# ============================================================
# GCN baseline
# ============================================================
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

try:
    model = GCN(X_torch.shape[1], 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = model(X_torch, edge_index_torch)
        loss = F.cross_entropy(out, torch.tensor(y_true))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        emb_gcn = model(X_torch, edge_index_torch).cpu().numpy()

    clustering = AgglomerativeClustering(
        n_clusters=num_classes, linkage="ward"
    )
    y_pred = clustering.fit_predict(emb_gcn)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    RES_ari["GCN"].append(ari)
    RES_nmi["GCN"].append(nmi)

    print(f"GCN -> ARI: {ari:.3f}, NMI: {nmi:.3f}")

except Exception as e:
    print(f"GCN failed: {e}")

# -----------------------------
# Summary
# -----------------------------
for method in RES_ari:
    print(
        f"\n{method}: "
        f"ARI mean={np.nanmean(RES_ari[method]):.3f}, "
        f"NMI mean={np.nanmean(RES_nmi[method]):.3f}"
    )