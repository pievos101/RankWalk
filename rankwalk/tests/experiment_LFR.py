# rankwalk/tests/experiment_LFR.py
import torch
import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rankwalk.conv import RankWalkConv
from node2vec import Node2Vec  # pip install node2vec

# -----------------------------
# Simulation parameters
# -----------------------------
n_iter = 20
out_dim = 64          # embedding dimension
walk_length = 20      # RankWalk walk length
use_node_features = False  # If True, use extra node features; else identity matrix

# Store results
RES_ari = {"RankWalk": [], "Node2Vec": []}
RES_nmi = {"RankWalk": [], "Node2Vec": []}

# -----------------------------
# Simulation loop
# -----------------------------
for it in range(1, n_iter + 1):
    print(f"\nIteration {it}/{n_iter}")

    # LFR benchmark parameters
    n_nodes = 100
    tau1 = 2.0
    tau2 = 1.1
    mu = 0.05
    min_degree = 5
    max_degree = 20
    min_community = 5
    max_community = 50

    try:
        G = nx.LFR_benchmark_graph(
            n=n_nodes,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            min_degree=min_degree,
            max_degree=max_degree,
            min_community=min_community,
            max_community=max_community,
            seed=it
        )
    except nx.ExceededMaxIterations:
        print("Skipping iteration due to LFR generation failure")
        continue

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    n_nodes_actual = G.number_of_nodes()
    if n_nodes_actual == 0:
        print("No nodes left after removing isolates, skipping iteration")
        continue

    # -----------------------------
    # Ground-truth communities
    # -----------------------------
    # Pick first community if node belongs to multiple (disjoint evaluation)
    y_true = np.array([list(G.nodes[n]['community'])[0] for n in G.nodes()])

    # Determine number of communities from ground truth
    unique_communities = sorted(set(y_true))
    num_communities = len(unique_communities)
    print(f"Number of true communities: {num_communities}")
    for c in unique_communities:
        print(f"Community {c}: {(y_true == c).sum()} nodes")

    # -----------------------------
    # Node features
    # -----------------------------
    if use_node_features:
        # Example: random features per node
        X_torch = torch.randn(n_nodes_actual, out_dim)
    else:
        X_torch = torch.eye(n_nodes_actual, dtype=torch.float)

    # -----------------------------
    # Edge index for RankWalk (PyG-style)
    # -----------------------------
    edges = np.array(list(G.edges())).T
    # Make bidirectional
    edge_index = torch.tensor(np.hstack([edges, edges[::-1]]), dtype=torch.long)

    # -----------------------------
    # RankWalk embeddings
    # -----------------------------
    in_dim = X_torch.shape[1]
    conv = RankWalkConv(in_dim, out_dim, walk_length=walk_length)
    try:
        embeddings_rw = conv(X_torch, edge_index)
        embeddings_rw_np = embeddings_rw.detach().numpy()

        clustering_rw = AgglomerativeClustering(
            n_clusters=num_communities, linkage="ward"
        )
        y_pred_rw = clustering_rw.fit_predict(embeddings_rw_np)

        ari_rw = adjusted_rand_score(y_true, y_pred_rw)
        nmi_rw = normalized_mutual_info_score(y_true, y_pred_rw)

        RES_ari["RankWalk"].append(ari_rw)
        RES_nmi["RankWalk"].append(nmi_rw)
        print(f"RankWalk -> ARI: {ari_rw:.3f}, NMI: {nmi_rw:.3f}")
    except Exception as e:
        print(f"RankWalk failed: {e}")

    # -----------------------------
    # Node2Vec embeddings (pip package)
    # -----------------------------
    try:
        node2vec_model = Node2Vec(
            G, dimensions=out_dim, walk_length=walk_length,
            num_walks=20, workers=1, seed=it
        )
        n2v_embeddings = node2vec_model.fit(window=10, min_count=1, batch_words=4)

        embeddings_n2v_np = np.array([n2v_embeddings.wv[str(n)] for n in G.nodes()])

        clustering_n2v = AgglomerativeClustering(
            n_clusters=num_communities, linkage="ward"
        )
        y_pred_n2v = clustering_n2v.fit_predict(embeddings_n2v_np)

        ari_n2v = adjusted_rand_score(y_true, y_pred_n2v)
        nmi_n2v = normalized_mutual_info_score(y_true, y_pred_n2v)

        RES_ari["Node2Vec"].append(ari_n2v)
        RES_nmi["Node2Vec"].append(nmi_n2v)
        print(f"Node2Vec -> ARI: {ari_n2v:.3f}, NMI: {nmi_n2v:.3f}")
    except Exception as e:
        print(f"Node2Vec failed: {e}")

# -----------------------------
# Summary
# -----------------------------
for method in ["RankWalk", "Node2Vec"]:
    ari_vals = RES_ari[method]
    nmi_vals = RES_nmi[method]
    print(f"\n{method}: ARI mean={np.nanmean(ari_vals):.3f}, NMI mean={np.nanmean(nmi_vals):.3f}")