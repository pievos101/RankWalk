import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# Restore the relative imports for your contrastive loss functions
from .contrastive import sample_pos_pairs_start_anchor, contrastive_loss_weighted_fixed

class FeatureAwareGNN(MessagePassing):
    def __init__(self, in_dim, num_nodes, hidden_dim=64, out_dim=48):
        super().__init__(aggr='add')
        
        # Expressive feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dedicated pure structural encoder (Uses structural ID embeddings)
        self.struct_embed = nn.Embedding(num_nodes, hidden_dim)
        self.struct_encoder = nn.Linear(hidden_dim, hidden_dim)
        
        # Interaction Fusion Layer
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.out = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, node_ids=None):
        if node_ids is None:
            node_ids = torch.arange(x.size(0), device=x.device)
            
        #h_feat = self.feat_encoder(x + 0.01 * torch.randn_like(x))
        h_feat = self.feat_encoder(x)
        
        s_init = self.struct_embed(node_ids)
        h_struct = self.propagate(edge_index, x=s_init)
        h_struct = self.struct_encoder(h_struct)
        
        h = self.fuse(torch.cat([h_feat, h_struct], dim=1))
        h = self.norm(h)
        h = self.out(h)
        
        return F.normalize(h, dim=1)

    def message(self, x_j):
        return x_j


def train_gnn(x, edge_index, J, epochs=500, lr=1e-3, walk_length=10, top_k=5, device='cpu'):
    x, edge_index, J = x.to(device), edge_index.to(device), J.to(device)
    
    model = FeatureAwareGNN(in_dim=x.size(1), num_nodes=x.size(0), out_dim=48).to(device)
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