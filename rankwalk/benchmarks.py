import torch
from torch_geometric.utils import from_networkx, to_undirected
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from node2vec import Node2Vec
import numpy as np

# Use package-relative imports
try:
    from .data import generate_sbm_graph
    from .utils import compute_jaccard_fast
    from .gnn import train_gnn
    from .contrastive import sample_pos_pairs_start_anchor, contrastive_loss_weighted_fixed
except ImportError:
    # fallback for direct run
    from data import generate_sbm_graph
    from utils import compute_jaccard_fast
    from gnn import train_gnn
    from contrastive import sample_pos_pairs_start_anchor, contrastive_loss_weighted_fixed

import networkx as nx

# --------------------------
# Node2Vec baseline
# --------------------------
def run_node2vec(G, dim=48):
    node2vec = Node2Vec(
        G,
        dimensions=dim,
        walk_length=30,
        num_walks=100,
        p=1,
        q=1,
        workers=1,
        seed=42
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=128)

    emb = torch.zeros(G.number_of_nodes(), dim)
    for i in range(G.number_of_nodes()):
        emb[i] = torch.tensor(model.wv[str(i)])
    return emb

# --------------------------
# Evaluation
# --------------------------
def evaluate(emb, labels):
    emb = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else emb
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    k = len(set(labels))
    km = KMeans(n_clusters=k, n_init=10).fit(emb)
    ari = adjusted_rand_score(labels, km.labels_)
    nmi = normalized_mutual_info_score(labels, km.labels_)
    return ari, nmi

# --------------------------
# Benchmark sweep
# --------------------------
def sbm_pout_sweep(p_out_values=None, n_repeats=5):
    if p_out_values is None:
        p_out_values = [0.05, 0.1, 0.15, 0.2]

    print("===== SBM p_out sweep =====")
    print(" SBM p_out | StartAnchor ARI ± std | Node2Vec ARI ± std")
    print("-------------------------------------------------------")

    for p_out in p_out_values:
        ari_gnn_list = []
        ari_n2v_list = []

        for _ in range(n_repeats):
            G, labels = generate_sbm_graph(
                n_communities=3, size_per_comm=30, p_in=0.4, p_out=p_out
            )
            data = from_networkx(G)
            edge_index = to_undirected(data.edge_index)

            x = torch.randn(G.number_of_nodes(), 20)
            J = compute_jaccard_fast(edge_index, G.number_of_nodes())

            emb_gnn = train_gnn(x, edge_index, J, epochs=100, walk_length=15, top_k=10)
            ari_gnn, _ = evaluate(emb_gnn, labels)
            ari_gnn_list.append(ari_gnn)

            emb_n2v = run_node2vec(G)
            ari_n2v, _ = evaluate(emb_n2v, labels)
            ari_n2v_list.append(ari_n2v)

        print(f" {p_out:6.2f} | {np.mean(ari_gnn_list):.3f} ± {np.std(ari_gnn_list):.3f} | "
              f"{np.mean(ari_n2v_list):.3f} ± {np.std(ari_n2v_list):.3f}")

# --------------------------
# General multi-benchmark run
# --------------------------
def run_all_benchmarks():
    benchmarks = {
        "SBM p_out sweep": sbm_pout_sweep,
        # Add more sweeps: DC-SBM, Ring-of-Cliques, RGG, Hierarchical SBM
    }
    for name, func in benchmarks.items():
        print(f"\n=== {name} ===")
        func()

# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    print("=== Running RankWalk Benchmarks ===")
    run_all_benchmarks()