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
        num_relations=2,
        cache_graph=False
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

        self.conv1 = GCNConv(hidden_dim, hidden_dim, cached=cache_graph)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, cached=cache_graph)

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
    device='cpu',
    *,
    backend='torch',
    seed=None,
    cache_graph=True,
    verbose=True,
):
    """Train the existing GCN and return detached Torch node embeddings.

    Existing positional arguments, default Torch sampling and device behavior are
    retained. ``backend='numba'`` is an optional CPU-only sampler and accepts dense
    Torch or SciPy CSR similarities. Install ``rankwalk[fast]`` to enable it.
    ``seed`` seeds Torch (including model initialization) and the fast sampler;
    otherwise normal global Torch randomness is retained. Fast and Torch backends
    use different random streams and are not bit-for-bit interchangeable.

    Normalization is cached only within this call's fixed graph. Standalone
    FeatureAwareRGNN instances remain uncached by default. ``edge_type`` is
    accepted as before; the current GCN architecture does not consume it.
    """
    from .contrastive import _weighted_loss
    from .utils import _validate_edges
    if backend not in ('torch', 'numba'):
        raise ValueError("backend must be 'torch' or 'numba'")
    target = torch.device(device)
    if backend == 'numba' and target.type != 'cpu':
        raise ValueError("backend='numba' is CPU-only; use backend='torch' for CUDA")
    if epochs < 1 or lr <= 0 or walk_length < 0 or top_k < 1:
        raise ValueError('epochs and lr must be positive; walk_length >= 0 and top_k >= 1')
    if x.ndim != 2 or x.shape[0] == 0 or not torch.is_floating_point(x):
        raise ValueError('x must be a non-empty floating-point node-feature matrix')
    _validate_edges(edge_index, len(x))
    if seed is not None:
        torch.manual_seed(seed)
    x, edge_index = x.to(target), edge_index.to(target, dtype=torch.long)
    if backend == 'numba':
        from ._fast import CPUWalkSampler
        sampler = CPUWalkSampler(edge_index, J, len(x), seed=seed)
    else:
        if not isinstance(J, torch.Tensor) or J.layout != torch.strided:
            raise TypeError("The Torch backend requires dense Torch J; use the Numba backend for CSR J")
        if J.shape != (len(x), len(x)):
            raise ValueError('J must have shape (num_nodes, num_nodes)')
        J = J.to(target)
    model = FeatureAwareRGNN(in_dim=x.size(1), out_dim=48,
                             cache_graph=cache_graph).to(device=target, dtype=x.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_emb, best_loss = None, float('inf')
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        emb = model(x, edge_index, edge_type)
        if backend == 'numba':
            pairs = sampler.sample(walk_length, top_k)
            i = torch.from_numpy(pairs[:, 0].astype('int64'))
            j = torch.from_numpy(pairs[:, 1].astype('int64'))
            weights = torch.as_tensor(pairs[:, 2], dtype=emb.dtype)
            loss = _weighted_loss(emb, i, j, weights)
        else:
            pairs = sample_pos_pairs_start_anchor(J, edge_index, len(x), walk_length, top_k)
            loss = contrastive_loss_weighted_fixed(emb, pairs)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite contrastive loss; check features and similarities')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        value = loss.item()
        if value < best_loss:
            best_loss, best_emb = value, emb.detach().clone()
        if verbose and epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | InfoNCE Loss: {value:.4f}')
    return best_emb
