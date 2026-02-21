import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import networkx as nx
import numpy as np

# --- Load Karate Club Graph ---
G = nx.karate_club_graph()
x = torch.eye(G.number_of_nodes(), dtype=torch.float)  # identity features
edge_index = torch.tensor(list(G.edges)).t().contiguous()
edge_index = to_undirected(edge_index)

labels = torch.tensor([0 if G.nodes[i]['club']=='Mr. Hi' else 1 for i in G.nodes()])

print(f"Downsampled graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Number of classes: {len(labels.unique())}")

# --- GNN Model ---
class TopKGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)) + h1)  # residual
        out = self.conv3(h2, edge_index)
        return out

# --- Training ---
def train(model, x, edge_index, epochs=500, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        embeddings = model(x, edge_index)
        # simple self-supervised: push all embeddings to be close to neighbors
        row, col = edge_index
        pos_loss = F.mse_loss(embeddings[row], embeddings[col])
        loss = pos_loss
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return embeddings.detach()

# --- Evaluation ---
def evaluate(embeddings, y):
    X = embeddings.numpy()
    y = y.numpy()
    kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=10).fit(X)
    ward = AgglomerativeClustering(n_clusters=len(np.unique(y))).fit(X)
    print("TopKGraphs Self-Supervised Embeddings:")
    print(f"-> KMeans ARI: {adjusted_rand_score(y, kmeans.labels_):.3f}, NMI: {normalized_mutual_info_score(y, kmeans.labels_):.3f}")
    print(f"-> Ward   ARI: {adjusted_rand_score(y, ward.labels_):.3f}, NMI: {normalized_mutual_info_score(y, ward.labels_):.3f}")

# --- Main ---
hidden_dim = 64
out_dim = 16
model = TopKGraph(x.size(1), hidden_dim, out_dim)
embeddings = train(model, x, edge_index, epochs=500, lr=0.01)
evaluate(embeddings, labels)