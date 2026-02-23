import torch
from torch_geometric.utils import from_networkx, to_undirected
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rankwalk import compute_jaccard_fast, train_gnn
from node2vec import Node2Vec
import networkx as nx
import numpy as np

# --------------------------
# Robust LFR generator
# --------------------------
def generate_lfr_graph(
    n=250,
    tau1=2.5,
    tau2=1.5,
    mu=0.15,
    avg_degree=10,
    min_community=10,
    max_tries=10,
    seed=42
):
    for attempt in range(max_tries):
        try:
            G = nx.LFR_benchmark_graph(
                n=n,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=avg_degree,
                min_community=min_community,
                seed=seed + attempt
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
        workers=1,
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
# Experiment parameters
# --------------------------
#n_nodes = 100
#avg_degree = 5
#max_degree = 10
#min_community <- 10
#max_community = 50
#mu = 0.30 # Mixing parameter
#tau1 = 2 #2
#tau2 = 1.1 #1.1   #

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
# Run 10 iterations
# --------------------------
ari_gnn_list, nmi_gnn_list = [], []
ari_n2v_list, nmi_n2v_list = [], []

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
        #seed=42 + i
    )

    #data = from_networkx(G)
    #edge_index = to_undirected(data.edge_index)
    #x = torch.randn(G.number_of_nodes(), 20)  # random features
    #J = compute_jaccard_fast(edge_index, G.number_of_nodes())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = from_networkx(G)
    # Move the WHOLE data object (edges + attributes) to GPU    
    data = data.to(device)
    edge_index = to_undirected(data.edge_index)
    # Create node features directly on GPU
    x = torch.randn(G.number_of_nodes(), 20, device=device)
    # Make sure compute_jaccard_fast runs on GPU tensors
    J = compute_jaccard_fast(edge_index, G.number_of_nodes())


    # Train StartAnchor GNN
    emb_gnn = train_gnn(
        x,
        edge_index,
        J,
        epochs=epochs,
        lr=lr,
        walk_length=walk_length,
        top_k=top_k
    )
    ari, nmi = evaluate(emb_gnn, labels)
    ari_gnn_list.append(ari)
    nmi_gnn_list.append(nmi)

    # Node2Vec baseline
    emb_n2v = run_node2vec(G, walk_length=walk_length)
    ari, nmi = evaluate(emb_n2v, labels)
    ari_n2v_list.append(ari)
    nmi_n2v_list.append(nmi)

# --------------------------
# Summary
# --------------------------
print("\n=== LFR Benchmark Summary over 10 iterations ===")
print(f"StartAnchor GNN | ARI: {np.mean(ari_gnn_list):.3f} ± {np.std(ari_gnn_list):.3f} | "
      f"NMI: {np.mean(nmi_gnn_list):.3f} ± {np.std(nmi_gnn_list):.3f}")
print(f"Node2Vec        | ARI: {np.mean(ari_n2v_list):.3f} ± {np.std(ari_n2v_list):.3f} | "
      f"NMI: {np.mean(nmi_n2v_list):.3f} ± {np.std(nmi_n2v_list):.3f}")