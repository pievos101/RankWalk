import torch
from torch_geometric.utils import from_networkx, to_undirected
from rankwalk import compute_jaccard_fast, train_gnn
from node2vec import Node2Vec
import networkx as nx
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------
# Robust LFR generator
# --------------------------
def generate_lfr_graph(
    n=250,
    tau1=2.5,
    tau2=1.5,
    mu=0.30,
    avg_degree=10,
    min_community=10,
    max_tries=20,
):
    for attempt in range(max_tries):
        try:
            G = nx.LFR_benchmark_graph(
                n=n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=avg_degree,
                min_community=min_community
            )
            G.remove_edges_from(nx.selfloop_edges(G))

            # Extract community labels
            communities = {}
            labels = np.zeros(G.number_of_nodes(), dtype=int)
            label_id = 0
            for node, comms in G.nodes(data="community"):
                c = list(comms)[0]
                if c not in communities:
                    communities[c] = label_id
                    label_id += 1
                labels[node] = communities[c]

            return G, torch.tensor(labels, dtype=torch.long)
        except nx.ExceededMaxIterations:
            continue

    raise RuntimeError(
        "LFR generation failed after multiple attempts. "
        "Try lowering avg_degree or min_community."
    )

# --------------------------
# Node2Vec baseline
# --------------------------
def run_node2vec(G, dim=48, walk_length=20):
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

    emb = torch.zeros(G.number_of_nodes(), dim)
    for i in range(G.number_of_nodes()):
        emb[i] = torch.tensor(model.wv[str(i)])
    return emb

# --------------------------
# kNN-based evaluation
# --------------------------
def evaluate_knn(emb, labels, test_size=0.5, k=5, random_state=None):
    emb_np = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else emb
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    train_idx, test_idx, y_train, y_test = train_test_split(
        np.arange(len(labels_np)), labels_np, test_size=test_size, random_state=random_state
    )

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(emb_np[train_idx], y_train)
    y_pred = knn.predict(emb_np[test_idx])

    acc = accuracy_score(y_test, y_pred)
    return acc

# --------------------------
# Experiment parameters
# --------------------------
n_iter = 10
n = 100
avg_degree = 5
min_community = 10
tau1 = 2
tau2 = 1.1
mu = 0.05

walk_length = 20
top_k = 20
epochs = 300
lr = 1e-3

# --------------------------
# Run experiment
# --------------------------
acc_gnn_list, acc_n2v_list = [], []

for i in range(n_iter):
    print(f"\nIteration {i+1}/{n_iter}")

    # Generate LFR graph
    G, labels = generate_lfr_graph(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        avg_degree=avg_degree,
        min_community=min_community,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = from_networkx(G)
    data = data.to(device)
    edge_index = to_undirected(data.edge_index)

    # --------------------------
    # Node features: unsupervised
    # --------------------------
    # Option 1: random features
    x = torch.randn(G.number_of_nodes(), 20, device=device)

    # Optionally, you could also concatenate structural features:
    # degrees = torch.tensor([G.degree[i] for i in range(G.number_of_nodes())], dtype=torch.float).unsqueeze(1)
    # clust_coeff = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float).unsqueeze(1)
    # pagerank = torch.tensor(list(nx.pagerank(G).values()), dtype=torch.float).unsqueeze(1)
    # x = torch.cat([x, degrees.to(device), clust_coeff.to(device), pagerank.to(device)], dim=1)

    # Jaccard for StartAnchor GNN
    J = compute_jaccard_fast(edge_index, G.number_of_nodes())

    # --------------------------
    # Train StartAnchor GNN
    # --------------------------
    emb_gnn = train_gnn(
        x,
        edge_index,
        J,
        epochs=epochs,
        lr=lr,
        walk_length=walk_length,
        top_k=top_k
    )

    acc_gnn = evaluate_knn(emb_gnn, labels, test_size=0.5, k=5, random_state=i)
    acc_gnn_list.append(acc_gnn)

    # --------------------------
    # Node2Vec baseline
    # --------------------------
    emb_n2v = run_node2vec(G, walk_length=walk_length)
    acc_n2v = evaluate_knn(emb_n2v, labels, test_size=0.5, k=5, random_state=i)
    acc_n2v_list.append(acc_n2v)

# --------------------------
# Summary
# --------------------------
print("\n=== LFR Benchmark Summary (kNN node classification) ===")
print(f"StartAnchor GNN | Accuracy: {np.mean(acc_gnn_list):.3f} ± {np.std(acc_gnn_list):.3f}")
print(f"Node2Vec        | Accuracy: {np.mean(acc_n2v_list):.3f} ± {np.std(acc_n2v_list):.3f}")