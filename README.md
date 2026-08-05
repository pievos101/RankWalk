# RankWalk
<p align="center">
<img src="https://github.com/pievos101/RankWalk/blob/main/RankWalk_Logo.png" width="600">
</p>

**RankWalk** is a Python library for **node representation learning** using start-node anchored random walks on graphs, combined with contrastive learning via a lightweight GNN. 

---

## Features

- Compute **start-node anchored Jaccard similarity** for graph nodes.
- Train a **StartAnchor GNN** using **contrastive InfoNCE loss** on positive pairs derived from biased random walks.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/pievos101/RankWalk.git
cd RankWalk

# Optional: create and activate a virtual environment
python3 -m venv rankwalk-venv
source rankwalk-venv/bin/activate  # Linux/macOS
# .\rankwalk-venv\Scripts\activate  # Windows

# Install the package 
pip install -e .
```

## Example Usage on Graphs

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
    x=x, 
    edge_index=edge_index,
    edge_type=None,
    J=J,
    epochs=epochs,
    lr=lr,
    walk_length=walk_length,
    top_k=top_k
)

print("RankWalk embedding shape:", emb_gnn.shape)

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
        workers=1#,
        #seed=42
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
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    emb_np = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else emb
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    k = len(set(labels_np))
    km = KMeans(n_clusters=k, n_init=10).fit(emb_np)
    ari = adjusted_rand_score(labels_np, km.labels_)
    nmi = normalized_mutual_info_score(labels_np, km.labels_)
    return ari, nmi

ari_gnn, nmi_gnn = evaluate(emb_gnn, labels)
ari_n2v, nmi_n2v = evaluate(emb_n2v, labels)

print(f"RankWalk | ARI: {ari_gnn:.3f}, NMI: {nmi_gnn:.3f}")
print(f"Node2Vec | ARI: {ari_n2v:.3f}, NMI: {nmi_n2v:.3f}")
```

## Example Usage: Longitudinal Multivariate Tabular Data

Let us simulate some longitudinal data using the R-package [TAPIO](https://github.com/pievos101/TAPIO):

### Simulated Longitudinal Data

```R
library(TAPIO)

df <- TAPIO::simLongData(
    ranTimes = FALSE,
    n_i = 10,
    eta = 3,
    sigma_diag = c(5,5,5,5,5)
  )

write.table(df, file="df.txt")
```

The above code generates data consisting of five longitudinal variables.

### Converting Tabular Data to Temporal Graph
```python
import numpy as np
import pandas as pd
from rankwalk import build_temporal_graph

df = pd.read_table("df.txt", sep=r"\s+")
print(df.head(20))

G, labels_df = build_temporal_graph(df, k_similarity=5)
```

#### The Case of Irregular Time Measures

```R
library(TAPIO)

df <- TAPIO::simLongData(
    ranTimes = TRUE, # irregular time
    n_i = 10,
    eta = 3,
    sigma_diag = c(5,5,5,5,5)
  )

write.table(df, file="df.txt")
```
In case of irregular time we apply a sliding window approach:

```python
import numpy as np
import pandas as pd
from rankwalk import build_temporal_graph_grid

df = pd.read_table("df.txt", sep=r"\s+")
print(df.head(20))

G, labels_df = build_temporal_graph_grid(
        df,
        k_similarity=5,
        n_bins=5,
        overlap=0.5  
    )
```

### After Graph Construction we Enrich Nodes with Longitudinal Values

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

node_list = list(G.nodes())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_list, t_list = [], []

for n in node_list:
    x_list.append(G.nodes[n]['features'])
    t_list.append(G.nodes[n]['time'])

x = torch.tensor(np.array(x_list), dtype=torch.float32, device=device)

edges, et = [], []

for u, v, a in G.edges(data=True):
    edges.append([u, v])
    et.append(a['edge_type'])
    edges.append([v, u])
    et.append(a['edge_type'])

edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
edge_type = torch.tensor(et, dtype=torch.long, device=device)
```

### Run RankWalk on Temporal Graph

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from rankwalk import train_gnn, compute_jaccard_fast

J = compute_jaccard_fast(edge_index, G.number_of_nodes(), device=device)

epochs=100
lr=1e-3
top_k=5
walk_length=10

emb = train_gnn(
    x, edge_index, edge_type, J,
    epochs=epochs,
    lr=lr,
    walk_length=walk_length,
    top_k=top_k,
    device=device
)

embeddings = emb.detach().cpu().numpy()
subjects =  np.array([G.nodes[n]['subject'] for n in node_list])
print("RankWalk embedding shape:", embeddings.shape)

```

Note, subject-wise mean pooling is suggested before clustering. 

```python
unique_subjects = np.unique(subjects)

subject_embeddings = np.vstack([
    embeddings[subjects == s].mean(axis=0)
    for s in unique_subjects
])

print(subject_embeddings.shape)
```
Let's apply kmeans clustering:

```python
from sklearn.cluster import KMeans

k = 4  # number of clusters

km = KMeans(
    n_clusters=k,
    n_init=20
)

labels = km.fit_predict(subject_embeddings)
```

Let us check the clustering performance:

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# One true label per subject
true_labels = (
    df.groupby("subject")["cluster"]
      .first()
      .to_numpy()
)

ari = adjusted_rand_score(true_labels, labels)
nmi = normalized_mutual_info_score(true_labels, labels)

print(f"ARI = {ari:.3f}")
print(f"NMI = {nmi:.3f}")

```

## Citation

If you find {RankWalk} useful please cite [the paper](https://arxiv.org/abs/2607.25609):

```
@article{pfeifer2026contrastive,
  title={Contrastive Representation Learning of Longitudinal Disease Trajectories on Temporal Graphs},
  author={Pfeifer, Bastian},
  journal={arXiv preprint arXiv:2607.25609},
  year={2026}
}

```
