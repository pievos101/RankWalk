import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from rankwalk.conv import RankWalkConv

# -----------------------------
# 1️⃣ Generate synthetic node features and cluster labels
# -----------------------------
n_nodes = 50
n_features = 5
n_clusters = 3
#random_state = 42

X, y_true = make_blobs(
    n_samples=n_nodes, n_features=n_features, centers=n_clusters
)
X = torch.tensor(X, dtype=torch.float)
y_true = np.array(y_true)

# -----------------------------
# 2️⃣ Create a kNN graph from features
# -----------------------------
k = 5
A = kneighbors_graph(X.numpy(), n_neighbors=k, include_self=False)
edge_index = np.vstack(A.nonzero())
# node2vec package expects networkx graph
import networkx as nx
G = nx.Graph()
G.add_edges_from(edge_index.T)

# -----------------------------
# 3️⃣ RankWalkConv embeddings
# -----------------------------
in_channels = n_features
out_channels = 64
walk_length = 20

conv = RankWalkConv(in_channels, out_channels, walk_length=walk_length)
embeddings = conv(X, torch.tensor(edge_index, dtype=torch.long))
print("RankWalkConv embeddings shape:", embeddings.shape)

embeddings_np = embeddings.detach().numpy()
clustering_rw = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
y_pred_rw = clustering_rw.fit_predict(embeddings_np)

ari_rw = adjusted_rand_score(y_true, y_pred_rw)
nmi_rw = normalized_mutual_info_score(y_true, y_pred_rw)
print(f"RankWalkConv -> ARI: {ari_rw:.3f}, NMI: {nmi_rw:.3f}")

# -----------------------------
# 4️⃣ Node2Vec (pip package) embeddings
# -----------------------------
from node2vec import Node2Vec as Node2Vec_PyPI

# Create node2vec model
node2vec = Node2Vec_PyPI(
    G,
    dimensions=64,
    walk_length=walk_length,
    num_walks=10,
    workers=1,
    p=1.0,
    q=1.0
)

# Fit embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Extract embeddings in same node order
embeddings_n2v_np = np.array([model.wv[str(i)] for i in range(n_nodes)])

clustering_n2v = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
y_pred_n2v = clustering_n2v.fit_predict(embeddings_n2v_np)

ari_n2v = adjusted_rand_score(y_true, y_pred_n2v)
nmi_n2v = normalized_mutual_info_score(y_true, y_pred_n2v)
print(f"Node2Vec (pip) -> ARI: {ari_n2v:.3f}, NMI: {nmi_n2v:.3f}")