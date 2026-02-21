import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_undirected
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Load the graph
dataset = KarateClub()
data = dataset[0]
x = data.x
edge_index = to_undirected(data.edge_index)
labels = data.y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, edge_index, labels = x.to(device), edge_index.to(device), labels.to(device)

# Model
class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        return F.normalize(h2, dim=1)  # normalize embeddings

model = GNN(x.size(1), 48).to(device)  # slightly larger hidden dim

# Contrastive-style loss using cosine similarity
def contrastive_loss(emb, edge_index):
    src, dst = edge_index
    pos_sim = (emb[src] * emb[dst]).sum(dim=1)  # cosine approx since embeddings are normalized
    pos_loss = (1 - pos_sim).mean()  # neighbors close

    # Sample negatives
    num_nodes = emb.size(0)
    neg_i = torch.randint(0, num_nodes, (len(src)*2,), device=device)
    neg_j = torch.randint(0, num_nodes, (len(dst)*2,), device=device)
    neg_sim = (emb[neg_i] * emb[neg_j]).sum(dim=1)
    neg_loss = F.relu(neg_sim).mean()  # push non-neighbors apart
    
    return pos_loss + neg_loss

# Training
def train(model, x, edge_index, epochs=500, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        emb = model(x, edge_index)
        loss = contrastive_loss(emb, edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model(x, edge_index).detach()

# Evaluation
def evaluate(emb, labels):
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    n_clusters = len(set(labels))
    
    km = KMeans(n_clusters=n_clusters, n_init=10).fit(emb)
    ward = AgglomerativeClustering(n_clusters=n_clusters).fit(emb)
    
    print("TopKGraphs Self-Supervised Embeddings:")
    print(f"-> KMeans ARI: {adjusted_rand_score(labels, km.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, km.labels_):.3f}")
    print(f"-> Ward   ARI: {adjusted_rand_score(labels, ward.labels_):.3f}, NMI: {normalized_mutual_info_score(labels, ward.labels_):.3f}")

# Run
embeddings = train(model, x, edge_index, epochs=10000, lr=0.01)
evaluate(embeddings, labels)