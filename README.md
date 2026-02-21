# RankWalk

**RankWalk** is a Python library for **node representation learning** using start-node anchored random walks on graphs, combined with contrastive learning via a lightweight GNN. It also provides baseline embeddings with **Node2Vec** for comparison. This library is useful for evaluating community structure, clustering, and general graph embedding tasks.

---

## Features

- Generate **synthetic graphs** (Stochastic Block Models) with controllable parameters.
- Compute **start-node anchored Jaccard similarity** for graph nodes.
- Train a **StartAnchor GNN** using **contrastive InfoNCE loss** on positive pairs derived from biased random walks.
- Generate **Node2Vec embeddings** for baseline comparison.
- Evaluate embeddings using clustering metrics: **ARI** (Adjusted Rand Index) and **NMI** (Normalized Mutual Information).

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