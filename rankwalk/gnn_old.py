import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# Restore the relative imports for your contrastive loss functions
from .contrastive import sample_pos_pairs_start_anchor, contrastive_loss_weighted_fixed

class FeatureAwareGNN(MessagePassing):
    def __init__(self, in_dim, num_nodes, hidden_dim=64, out_dim=48):
        super().__init__(aggr='add')
        
        # 1. Expressive feature encoder maps raw attributes to hidden space
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # REMOVED: self.struct_embed
        # REMOVED: self.struct_encoder
        
        # 2. Changed input dimension from (hidden_dim * 2) to (hidden_dim) 
        # since tracks are no longer concatenated side-by-side
        self.post_prop_dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.out = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, node_ids=None):
        # 3. Compute raw feature embeddings 
        h_feat = self.feat_encoder(x)
        
        # 4. CRUCIAL CHANGE: Message passing now runs directly over feature states
        # rather than isolated structural ID embeddings
        h_graph = self.propagate(edge_index, x=h_feat)
        
        # 5. Process the combined topology-feature matrix through post-processing layers
        h = self.post_prop_dense(h_graph)
        h = self.norm(h)
        h = self.out(h)
        
        return F.normalize(h, dim=1)

    def message(self, x_j):
        # x_j now explicitly represents the feature profiles of neighboring nodes
        return x_j


def train_gnn(x, edge_index, J, epochs=500, lr=1e-3, walk_length=10, top_k=5, device='cpu'):
    x, edge_index, J = x.to(device), edge_index.to(device), J.to(device)
    
    model = FeatureAwareGNN(in_dim=x.size(1), num_nodes=x.size(0), out_dim=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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