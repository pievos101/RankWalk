"""Optional CPU walk sampler. Imported only when backend='numba' is requested."""
import numpy as np
import scipy.sparse as sp
import torch
try:
    from numba import njit
except ImportError as exc:
    raise ImportError("The Numba backend requires rankwalk[fast]; install with pip install 'rankwalk[fast]'") from exc
from .utils import _validate_edges


@njit(cache=True)
def _lookup(indptr, indices, data, row, col):
    lo, end = indptr[row], indptr[row + 1]
    hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if indices[mid] < col:
            lo = mid + 1
        else:
            hi = mid
    if lo < end and indices[lo] == col:
        return data[lo]
    return 0.0


@njit(cache=True)
def _sample_pairs(ap, ai, counts, jp, ji, jv, uniforms, top_k, eps=1e-6):
    n, walk_length = uniforms.shape
    output = np.empty((n * max(0, top_k - 1), 3), dtype=np.float64)
    size = 0
    for start in range(n):
        visited = np.empty(walk_length + 1, dtype=np.int64)
        ranks = np.empty(walk_length + 1, dtype=np.int64)
        visited[0], ranks[0], used, current = start, 1, 1, start
        for step in range(walk_length):
            lo, hi = ap[current], ap[current + 1]
            if lo == hi:
                break
            total = 0.0
            for i in range(lo, hi):
                total += (_lookup(jp, ji, jv, start, ai[i]) + eps) * counts[i]
            cutoff, cumulative, following = uniforms[start, step] * total, 0.0, ai[hi - 1]
            for i in range(lo, hi):
                cumulative += (_lookup(jp, ji, jv, start, ai[i]) + eps) * counts[i]
                if cutoff < cumulative:
                    following = ai[i]
                    break
            current = following
            found = False
            for i in range(used):
                if visited[i] == current:
                    found = True
                    break
            if not found:
                visited[used], ranks[used] = current, step + 2
                used += 1
        stop = min(used, top_k)
        maximum = ranks[stop - 1]
        for i in range(1, stop):
            output[size, 0] = start
            output[size, 1] = visited[i]
            output[size, 2] = 1.0 - ranks[i] / (maximum + eps)
            size += 1
    return output[:size]


class CPUWalkSampler:
    """Prepare CSR data once; preserve directed walks and duplicate-edge mass."""
    def __init__(self, edge_index, similarity, num_nodes, seed=None):
        row, col = _validate_edges(edge_index, num_nodes)
        # Jaccard is undirected, but the original walk follows supplied directions.
        self.adjacency = sp.csr_matrix((np.ones(len(row), dtype=np.int64), (row, col)),
                                      shape=(num_nodes, num_nodes))
        self.adjacency.sort_indices()
        if isinstance(similarity, torch.Tensor):
            value = similarity.detach().cpu()
            value = value.to(torch.float64 if value.dtype == torch.float64 else torch.float32)
            if value.layout == torch.strided:
                matrix = sp.csr_matrix(value.numpy())
            else:
                value = value.to_sparse_coo().coalesce()
                indices = value.indices().numpy()
                matrix = sp.csr_matrix((value.values().numpy(), indices), shape=tuple(value.shape))
        elif sp.issparse(similarity):
            matrix = similarity.tocsr(copy=True)
        else:
            raise TypeError('J must be a Torch tensor or SciPy sparse matrix')
        if matrix.shape != (num_nodes, num_nodes):
            raise ValueError('J must have shape (num_nodes, num_nodes)')
        matrix = matrix.astype(np.float64 if matrix.dtype == np.float64 else np.float32)
        matrix.sum_duplicates()
        matrix.sort_indices()
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError('J must contain finite non-negative similarities')
        self.similarity = matrix
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)

    def sample(self, walk_length, top_k):
        if walk_length < 0 or top_k < 1:
            raise ValueError('walk_length must be non-negative and top_k at least 1')
        a, j = self.adjacency, self.similarity
        return _sample_pairs(a.indptr, a.indices, a.data, j.indptr, j.indices,
                             j.data, self.rng.random((self.num_nodes, walk_length)), top_k)
