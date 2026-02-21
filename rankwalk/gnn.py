import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .contrastive import sample_pos_pairs_start_anchor, contrastive_loss_weighted_fixed

class VanillaGNN(MessagePassing):
    def __init__(self, in_dim, hidden_dim=48, out_dim=48):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.lin1(x))
        h = self.propagate(edge_index, x=h)
        h = self.lin2(h)
        return F.normalize(h, dim=1)

    def message(self, x_j):
        return x_j

def train_gnn(x, edge_index, J, epochs=500, lr=1e-3, walk_length=10, top_k=5, device='cpu'):
    x, edge_index, J = x.to(device), edge_index.to(device), J.to(device)
    model = VanillaGNN(x.size(1), out_dim=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_emb = None
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index)
        pos_pairs = sample_pos_pairs_start_anchor(J, edge_index, x.size(0), walk_length, top_k)
        loss = contrastive_loss_weighted_fixed(emb, pos_pairs)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | InfoNCE Loss: {loss.item():.4f}")

    return best_emb