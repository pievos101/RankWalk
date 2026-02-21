import torch
from torch_geometric.utils import from_networkx, to_undirected
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rankwalk import generate_sbm_graph, compute_jaccard_fast, train_gnn

# --------------------------
# Generate graph and features
# --------------------------
G, labels = generate_sbm_graph()
data = from_networkx(G)
edge_index = to_undirected(data.edge_index)

x = torch.randn(G.number_of_nodes(), 20)  # random node features
J = compute_jaccard_fast(edge_index, G.number_of_nodes())

# --------------------------
# Train StartAnchor GNN
# --------------------------
emb = train_gnn(x, edge_index, J, epochs=100)
print("Embedding shape:", emb.shape)

# --------------------------
# Evaluate with KMeans clustering
# --------------------------
def evaluate_ari(emb, labels):
    emb_np = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else emb
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    k = len(set(labels_np))
    km = KMeans(n_clusters=k, n_init=10).fit(emb_np)
    ari = adjusted_rand_score(labels_np, km.labels_)
    nmi = normalized_mutual_info_score(labels_np, km.labels_)
    print(f"Clustering results | ARI: {ari:.3f}, NMI: {nmi:.3f}")
    return ari, nmi

ari, nmi = evaluate_ari(emb, labels)