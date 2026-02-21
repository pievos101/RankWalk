# RankWalk

**RankWalk** is a Python library for **node representation learning** using start-node anchored random walks on graphs, combined with contrastive learning via a lightweight GNN. It also provides baseline embeddings with **Node2Vec** for comparison. This library is useful for evaluating community structure, clustering, and general graph embedding tasks.

---

## Features

- Compute **start-node anchored Jaccard similarity** for graph nodes.
- Train a **StartAnchor GNN** using **contrastive InfoNCE loss** on positive pairs derived from biased random walks.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/RankWalk.git
cd RankWalk

# Optional: create and activate a virtual environment
python3 -m venv rankwalk-venv
source rankwalk-venv/bin/activate  # Linux/macOS
# .\rankwalk-venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

## Example Usage

```python
import torch
from torch_geometric.utils import from_networkx, to_undirected
from sklearn.cluster import KMeans
from rankwalk import generate_sbm_graph, compute_jaccard_fast, train_gnn
from node2vec import Node2Vec

# --------------------------
# Graph parameters
# --------------------------
n_communities = 4
size_per_comm = 25
p_in = 0.4
p_out = 0.15
seed = 42

# Generate SBM graph
G, labels = generate_sbm_graph(
    n_communities=n_communities,
    size_per_comm=size_per_comm,
    p_in=p_in,
    p_out=p_out,
    seed=seed
)

data = from_networkx(G)
edge_index = to_undirected(data.edge_index)

# --------------------------
# Node features
# --------------------------
x = torch.randn(G.number_of_nodes(), 20)  # random features
J = compute_jaccard_fast(edge_index, G.number_of_nodes())

# --------------------------
# Train StartAnchor GNN
# --------------------------
walk_length = 20
top_k = 10
epochs = 300
lr = 1e-3

emb_gnn = train_gnn(
    x, edge_index, J,
    epochs=epochs,
    lr=lr,
    walk_length=walk_length,
    top_k=top_k
)

print("StartAnchor GNN embedding shape:", emb_gnn.shape)

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
        seed=42
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=128)

    emb = torch.zeros(G.number_of_nodes(), dim)
    for i in range(G.number_of_nodes()):
        emb[i] = torch.tensor(model.wv[str(i)])
    return emb

emb_n2v = run_node2vec(G)
print("Node2Vec embedding shape:", emb_n2v.shape)

# --------------------------
# Evaluate embeddings
# --------------------------
def evaluate(emb, labels):
    emb_np = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else emb
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    k = len(set(labels_np))
    km = KMeans(n_clusters=k, n_init=10).fit(emb_np)
    ari = adjusted_rand_score(labels_np, km.labels_)
    nmi = normalized_mutual_info_score(labels_np, km.labels_)
    return ari, nmi

ari_gnn, nmi_gnn = evaluate(emb_gnn, labels)
ari_n2v, nmi_n2v = evaluate(emb_n2v, labels)

print(f"StartAnchor GNN | ARI: {ari_gnn:.3f}, NMI: {nmi_gnn:.3f}")
print(f"Node2Vec        | ARI: {ari_n2v:.3f}, NMI: {nmi_n2v:.3f}")
```
