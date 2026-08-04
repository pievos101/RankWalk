import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GCNConv

from .contrastive import (
    sample_pos_pairs_start_anchor,
    contrastive_loss_weighted_fixed,
    contrastive_loss_mixed_negatives
)


TEMPORAL_EDGE = 0
SIMILARITY_EDGE = 1

### !
### IN CASE OF NONTEMPORAL GRAPHS SWITCH TO GCNConv instead of RGCNConv!!
### !

class FeatureAwareRGNN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim=64,
        out_dim=48,
        num_relations=2
    ):
        super().__init__()

        # -----------------------------------------
        # Feature encoder
        # -----------------------------------------
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # -----------------------------------------
        # Relation-aware graph convolutions
        # -----------------------------------------
        #self.conv1 = RGCNConv(
        #    hidden_dim,
        #    hidden_dim,
        #    num_relations=num_relations
        #)

        #self.conv2 = RGCNConv(
        #    hidden_dim,
        #    hidden_dim,
        #    num_relations=num_relations
        #)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # -----------------------------------------
        # Post-processing
        # -----------------------------------------
        self.norm = nn.LayerNorm(hidden_dim)

        self.post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x,
        edge_index,
        edge_type
    ):

        # -----------------------------------------
        # Initial feature representation
        # -----------------------------------------
        h = self.feat_encoder(x)

        # Residual anchor
        h0 = h

        # -----------------------------------------
        # Relation-aware propagation
        # -----------------------------------------
        #h = self.conv1(h, edge_index, edge_type)
        #h = F.relu(h)

        #h = self.conv2(h, edge_index, edge_type)

        h = self.conv1(h, edge_index)
        h = F.relu(h)

        h = self.conv2(h, edge_index)

        # -----------------------------------------
        # Residual connection
        # -----------------------------------------
        h = h + h0

        # -----------------------------------------
        # Post-processing
        # -----------------------------------------
        h = self.norm(h)

        h = self.post(h)

        h = self.out(h)

        return F.normalize(h, dim=1)

def apply_edge_dropout(edge_index, edge_type, p=0.2):
    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

    sim_mask = (edge_type == SIMILARITY_EDGE)

    dropout_mask = torch.rand(sim_mask.sum(), device=edge_index.device) > p

    keep_mask[sim_mask] = dropout_mask

    return edge_index[:, keep_mask], edge_type[keep_mask]

def train_gnn(
    x,
    edge_index,
    edge_type,
    J,
    epochs=500,
    lr=1e-3,
    walk_length=10,
    top_k=5,
    device='cpu'
):

    x = x.to(device)
    edge_index = edge_index.to(device)
    #edge_type = edge_type.to(device)
    J = J.to(device)

    model = FeatureAwareRGNN(
        in_dim=x.size(1),
        out_dim=48
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best_emb = None
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        # -----------------------------------------
        # EDGE DROPOUT (HERE)
        # -----------------------------------------
        #edge_index_d, edge_type_d = apply_edge_dropout(
        #    edge_index,
        #    edge_type,
        #    p=0.1  # start small: 5–15%
        #)

        emb = model(
            x,
            edge_index,
            edge_type
        )

        pos_pairs = sample_pos_pairs_start_anchor(
            J,
            edge_index,
            x.size(0),
            walk_length,
            top_k
        )

        loss = contrastive_loss_weighted_fixed(
            emb,
            pos_pairs
        )

        #loss = contrastive_loss_mixed_negatives(
        #    emb,
        #    pos_pairs,
        #    J
        #)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )

        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_emb = emb.detach()

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"InfoNCE Loss: {loss.item():.4f}"
            )

    return best_emb