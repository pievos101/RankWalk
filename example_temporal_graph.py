import pandas as pd
import numpy as np
import torch
import networkx as nx

from torch_geometric.utils import from_networkx, to_undirected
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from node2vec import Node2Vec
from rankwalk import build_temporal_graph, train_gnn, compute_jaccard_fast


# =========================================================
# POOLING
# =========================================================
def pool_patient_embeddings(emb, node2patient, method="mean"):
    emb = emb.detach().cpu().numpy()
    node2patient = np.array(node2patient)

    unique_patients = np.unique(node2patient)
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}

    dim = emb.shape[1]
    patient_emb = np.zeros((len(unique_patients), dim))
    counts = np.zeros(len(unique_patients))

    for i, p in enumerate(node2patient):
        pi = patient_to_idx[p]
        patient_emb[pi] += emb[i]
        counts[pi] += 1

    if method == "mean":
        patient_emb = patient_emb / np.maximum(counts[:, None], 1)

    return torch.tensor(patient_emb, dtype=torch.float)


# =========================================================
# EVALUATION
# =========================================================
def evaluate(emb, labels):
    emb_np = emb.detach().cpu().numpy() if torch.is_tensor(emb) else emb
    labels_np = np.array(labels).ravel()

    k = len(np.unique(labels_np))
    km = KMeans(n_clusters=k, n_init=10)
    pred = km.fit_predict(emb_np)

    ari = adjusted_rand_score(labels_np, pred)
    nmi = normalized_mutual_info_score(labels_np, pred)

    return ari, nmi


# =========================================================
# NODE2VEC
# =========================================================
def run_node2vec(G, dim=48, walk_length=20):
    node_list = sorted(G.nodes())

    node2vec = Node2Vec(
        G,
        dimensions=dim,
        walk_length=walk_length,
        num_walks=100,
        p=1,
        q=1,
        workers=1
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=128)

    emb = torch.zeros((len(node_list), dim))

    for i, n in enumerate(node_list):
        emb[i] = torch.tensor(model.wv[str(n)])

    return emb


# =========================================================
# SETTINGS
# =========================================================
walk_length = 20
top_k = 10
epochs = 100
lr = 1e-3
n_iter = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# EXPERIMENT LOOP
# =========================================================
ari_gnn_list, nmi_gnn_list = [], []
ari_n2v_list, nmi_n2v_list = [], []


for it in range(n_iter):

    print(f"\n================ ITERATION {it+1}/{n_iter} ================")

    df = pd.read_csv("longData.csv")

    # --------------------------
    # Build graph
    # --------------------------
    G, labels_df = build_temporal_graph(df, k_similarity=10)

    node_list = sorted(G.nodes())
    labels = labels_df["cluster"].to_numpy().ravel()

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # =====================================================
    # 🔥 CRITICAL FIX: REMOVE EDGE ATTRIBUTES
    # =====================================================
    for u, v in G.edges():
        G[u][v].clear()

    # --------------------------
    # PyG conversion
    # --------------------------
    data = from_networkx(G)
    edge_index = to_undirected(data.edge_index).to(device)

    # --------------------------
    # Node features
    # --------------------------
    x = torch.tensor(
        np.array([G.nodes[n]["features"] for n in node_list]),
        dtype=torch.float,
        device=device
    )

    time_feat = torch.tensor(
        np.array([[G.nodes[n]["time"]] for n in node_list]),
        dtype=torch.float,
        device=device
    )

    x = torch.cat([x, time_feat], dim=1)

    # --------------------------
    # Jaccard
    # --------------------------
    J = compute_jaccard_fast(edge_index, G.number_of_nodes())

    # --------------------------
    # Train GNN
    # --------------------------
    emb_node = train_gnn(
        x,
        edge_index,
        J,
        epochs=epochs,
        lr=lr,
        walk_length=walk_length,
        top_k=top_k,
        device=device
    )

    # --------------------------
    # pooling
    # --------------------------
    node2patient = np.array([G.nodes[n]["subject"] for n in node_list])

    emb_patient_gnn = pool_patient_embeddings(emb_node, node2patient)

    ari, nmi = evaluate(emb_patient_gnn, labels)

    ari_gnn_list.append(ari)
    nmi_gnn_list.append(nmi)

    print(f"GNN   ARI={ari:.4f} | NMI={nmi:.4f}")

    # --------------------------
    # Node2Vec
    # --------------------------
    emb_node_n2v = run_node2vec(G, dim=48, walk_length=walk_length)

    emb_patient_n2v = pool_patient_embeddings(emb_node_n2v, node2patient)

    ari, nmi = evaluate(emb_patient_n2v, labels)

    ari_n2v_list.append(ari)
    nmi_n2v_list.append(nmi)

    print(f"N2V   ARI={ari:.4f} | NMI={nmi:.4f}")


# =========================================================
# SUMMARY
# =========================================================
print("\n================ FINAL RESULTS ================")

print(
    f"GNN | ARI {np.mean(ari_gnn_list):.3f} ± {np.std(ari_gnn_list):.3f} | "
    f"NMI {np.mean(nmi_gnn_list):.3f} ± {np.std(nmi_gnn_list):.3f}"
)

print(
    f"N2V | ARI {np.mean(ari_n2v_list):.3f} ± {np.std(ari_n2v_list):.3f} | "
    f"NMI {np.mean(nmi_n2v_list):.3f} ± {np.std(nmi_n2v_list):.3f}"
)