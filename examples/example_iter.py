import torch
from torch_geometric.utils import from_networkx, to_undirected
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rankwalk import generate_sbm_graph, compute_jaccard_fast, train_gnn
from node2vec import Node2Vec
import numpy as np

# --------------------------
# Graph parameters
# --------------------------
n_communities = 4
size_per_comm = 25
p_in = 0.5
p_out = 0.15
seed = 42

# --------------------------
# GNN training parameters
# --------------------------
walk_length = 20
top_k = 5
epochs = 500
lr = 1e-3

# --------------------------
# Node2Vec baseline
# --------------------------
def run_node2vec(G, dim=48):
    node2vec = Node2Vec(
        G,
        dimensions=dim,
        walk_length=20,
        num_walks=100,
        p=1,
        q=1,
        workers=1,
        #seed=42
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=128)

    emb = torch.zeros(G.number_of_nodes(), dim)
    for i in range(G.number_of_nodes()):
        emb[i] = torch.tensor(model.wv[str(i)])
    return emb

# --------------------------
# Evaluation function
# --------------------------
def evaluate(emb, labels):
    emb_np = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else emb
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    k = len(set(labels_np))
    km = KMeans(n_clusters=k, n_init=10).fit(emb_np)
    ari = adjusted_rand_score(labels_np, km.labels_)
    nmi = normalized_mutual_info_score(labels_np, km.labels_)
    return ari, nmi

# --------------------------
# Run multiple iterations
# --------------------------
n_iter = 10
ari_gnn_list, nmi_gnn_list = [], []
ari_n2v_list, nmi_n2v_list = [], []

for i in range(n_iter):
    print(f"\nIteration {i+1}/{n_iter}")
    
    # Generate new synthetic graph each iteration
    G, labels = generate_sbm_graph(
        n_communities=n_communities,
        size_per_comm=size_per_comm,
        p_in=p_in,
        p_out=p_out,
        seed=seed+i  # vary seed
    )

    data = from_networkx(G)
    edge_index = to_undirected(data.edge_index)
    x = torch.randn(G.number_of_nodes(), 20)
    J = compute_jaccard_fast(edge_index, G.number_of_nodes())

    # Train StartAnchor GNN
    emb_gnn = train_gnn(
        x, edge_index, J,
        epochs=epochs,
        lr=lr,
        walk_length=walk_length,
        top_k=top_k
    )
    ari, nmi = evaluate(emb_gnn, labels)
    ari_gnn_list.append(ari)
    nmi_gnn_list.append(nmi)

    # Node2Vec baseline
    emb_n2v = run_node2vec(G)
    ari, nmi = evaluate(emb_n2v, labels)
    ari_n2v_list.append(ari)
    nmi_n2v_list.append(nmi)

# --------------------------
# Summarize results
# --------------------------
print("\n=== Summary over 10 iterations ===")
print(f"StartAnchor GNN | ARI: {np.mean(ari_gnn_list):.3f} ± {np.std(ari_gnn_list):.3f} | "
      f"NMI: {np.mean(nmi_gnn_list):.3f} ± {np.std(nmi_gnn_list):.3f}")
print(f"Node2Vec        | ARI: {np.mean(ari_n2v_list):.3f} ± {np.std(ari_n2v_list):.3f} | "
      f"NMI: {np.mean(nmi_n2v_list):.3f} ± {np.std(nmi_n2v_list):.3f}")