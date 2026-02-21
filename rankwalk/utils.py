import torch

def compute_jaccard_fast(edge_index, num_nodes, device='cpu'):
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    row, col, data = [], [], []
    for u in range(num_nodes):
        Nu = neighbors[u]
        candidates = set()
        for v in Nu:
            candidates.update(neighbors[v])
        candidates.discard(u)
        for v in candidates:
            Nv = neighbors[v]
            union = Nu | Nv
            if union:
                row.append(u)
                col.append(v)
                data.append(len(Nu & Nv) / len(union))
    
    if len(row) == 0:
        return torch.zeros((num_nodes, num_nodes), device=device)
    
    J = torch.sparse_coo_tensor([row, col], data, (num_nodes, num_nodes), device=device).to_dense()
    return J