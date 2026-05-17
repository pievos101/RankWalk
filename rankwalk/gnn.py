import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .contrastive import sample_pos_pairs_start_anchor, contrastive_loss_weighted_fixed
from torch_geometric.utils import softmax

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

class FeatureAwareGNN_old(MessagePassing):
    def __init__(self, in_dim, hidden_dim=48, out_dim=48):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

        # Feature-dependent gate
        #self.gate = nn.Sequential(
        #    nn.Linear(hidden_dim, 1),
        #    nn.Sigmoid()
        #)

        # FEATURE-WISE GATING 
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, x, edge_index):
        h = F.relu(self.lin1(x))
        h = self.propagate(edge_index, x=h)
        return F.normalize(self.lin2(h), dim=1)

    def message(self, x_j):
        w = self.gate(x_j)
        return w * x_j

class FeatureAwareGNN(MessagePassing):
    def __init__(self, in_dim, hidden_dim=64, out_dim=48):
        super().__init__(aggr='add')

        # feature encoder (strong, expressive)
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # structural encoder (separate space!)
        self.struct_encoder = nn.Linear(hidden_dim, hidden_dim)

        # fusion network (learns interaction, not raw mixing)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.out = nn.Linear(hidden_dim, out_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):

        # subsample tests
        # feature drop out in one node
        #x = x * (torch.rand_like(x) > 0.1).float()
        # drop out of whole feature 
        #x = x * (torch.rand(1, x.size(1), device=x.device) > 0.1).float() 
        ###########

        #h_feat = self.feat_encoder(x)
        h_feat = self.feat_encoder(x + 0.01 * torch.randn_like(x))

        h_struct = self.propagate(edge_index, x=h_feat)
        h_struct = self.struct_encoder(h_struct)

        h = self.fuse(torch.cat([h_feat, h_struct], dim=1))

        h = self.norm(h)
        h = self.out(h)

        return F.normalize(h, dim=1)

    def message(self, x_j):
        return x_j

    def forward_multi_view(self, x, edge_index):

        # =====================================================
        # TAPIO-STYLE RANDOM FEATURE SUBSPACES
        # =====================================================

        p = 0.8

        mask1 = (
            torch.rand(1, x.size(1), device=x.device) < p
        ).float()

        mask2 = (
            torch.rand(1, x.size(1), device=x.device) < p
        ).float()

        # two stochastic feature views
        x1 = x * mask1
        x2 = x * mask2

        # =====================================================
        # SHARED FEATURE ENCODER
        # =====================================================

        h1 = self.feat_encoder(x1)
        h2 = self.feat_encoder(x2)

        # ensemble averaging (TAPIO consensus analogue)
        h_feat = (h1 + h2) / 2

        # =====================================================
        # GRAPH STRUCTURE
        # =====================================================

        h_struct = self.propagate(edge_index, x=h_feat)
        h_struct = self.struct_encoder(h_struct)

        # =====================================================
        # FUSION
        # =====================================================

        h = self.fuse(torch.cat([h_feat, h_struct], dim=1))

        h = self.norm(h)
        h = self.out(h)

        emb = F.normalize(h, dim=1)

        return emb, h1, h2

def train_gnn(x, edge_index, J, epochs=500, lr=1e-3, walk_length=10, top_k=5, device='cpu'):
    x, edge_index, J = x.to(device), edge_index.to(device), J.to(device)
    #model = VanillaGNN(x.size(1), out_dim=48).to(device)
    model = FeatureAwareGNN(x.size(1), out_dim=48).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_emb = None
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index)
        #emb, h1, h2 = model(x, edge_index)
        pos_pairs = sample_pos_pairs_start_anchor(J, edge_index, x.size(0), walk_length, top_k)
        loss = contrastive_loss_weighted_fixed(emb, pos_pairs)
        # TAPIO-style view consistency
        #loss_view = F.mse_loss(h1, h2)
        # combined loss
        #loss = loss_struct + 0.05 * loss_view
        #
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | InfoNCE Loss: {loss.item():.4f}")

    return best_emb

