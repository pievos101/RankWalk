"""Unweighted, undirected Jaccard similarity with optional sparse storage."""
from numbers import Integral
import numpy as np
import scipy.sparse as sp
import torch


def _validate_edges(edge_index, num_nodes):
    if not isinstance(num_nodes, Integral) or num_nodes < 0:
        raise ValueError('num_nodes must be a non-negative integer')
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must be a tensor with shape (2, num_edges)')
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError('edge_index must contain integer node indices')
    edges = edge_index.detach().cpu().numpy()
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError('edge_index contains an out-of-range node index')
    return edges


def compute_jaccard_sparse(edge_index, num_nodes):
    """Return a CPU SciPy CSR Jaccard matrix, without allocating a dense N×N array.

    As in the original implementation, edges are treated as undirected neighbor
    sets, duplicate edges do not change similarity, and the diagonal is zero.
    Sparse storage can still become quadratic for graphs with many two-hop pairs.
    """
    row, col = _validate_edges(edge_index, num_nodes)
    adjacency = sp.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)),
                              shape=(num_nodes, num_nodes))
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    common = (adjacency @ adjacency).tocoo()
    keep = common.row != common.col
    r, c, intersection = common.row[keep], common.col[keep], common.data[keep]
    union = degree[r] + degree[c] - intersection
    result = sp.csr_matrix((intersection / union, (r, c)), shape=adjacency.shape)
    result.sort_indices()
    return result


def compute_jaccard_fast(edge_index, num_nodes, device='cpu'):
    """Return the existing dense float32 Torch result on the requested device.

    Use ``compute_jaccard_sparse`` with the optional Numba training backend when
    dense storage is undesirable. The dense API and device argument are retained.
    """
    similarity = compute_jaccard_sparse(edge_index, num_nodes).tocoo()
    indices = torch.from_numpy(np.stack([similarity.row, similarity.col]).astype(np.int64))
    values = torch.from_numpy(similarity.data)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes),
                                   device=device).to_dense()
