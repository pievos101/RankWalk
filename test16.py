import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Load Karate Club graph
G = nx.karate_club_graph()
num_nodes = G.number_of_nodes()
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
labels = torch.tensor([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes], dtype=torch.long)

print(f"Downsampled graph: {num_nodes} nodes, {G.number_of_edges()} edges")
print(f"Number of classes: {len(torch.unique(labels))}")

# Random initial features (16-dim)
x = torch.randn(num_nodes, 16)

# Simple GCN-like layer
class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats)

    def forward(self, x, edge_index):
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])  # simple neighborhood aggregation
        agg = agg / (torch.bincount(row, minlength=x.size(0)).unsqueeze(1).float() + 1e-6)
        return F.relu(self.lin(agg) + x)  # residual

# Model
class GNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, hidden_feats))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_feats, hidden_feats))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

# Training function (graph contrastive-like)
def train(model, x, edge_index, epochs=500, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        embeddings = model(x, edge_index)

        # Self-supervised loss: neighbors should be similar
        row, col = edge_index
        loss = F.mse_loss(embeddings[row], embeddings[col])

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return embeddings

# Evaluation
def evaluate(embeddings, y):
    X = embeddings.detach().numpy()
    y = y.numpy()
    kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=10).fit(X)
    ward = AgglomerativeClustering(n_clusters=len(np.unique(y))).fit(X)

    print("\nTopKGraphs Self-Supervised Embeddings:")
    print(f"-> KMeans ARI: {adjusted_rand_score(y, kmeans.labels_):.3f}, NMI: {normalized_mutual_info_score(y, kmeans.labels_):.3f}")
    print(f"-> Ward   ARI: {adjusted_rand_score(y, ward.labels_):.3f}, NMI: {normalized_mutual_info_score(y, ward.labels_):.3f}")

# Initialize and train
model = GNN(in_feats=16, hidden_feats=16, num_layers=2)
embeddings = train(model, x, edge_index, epochs=500, lr=0.01)

# Evaluate
evaluate(embeddings, labels)