import torch

def sample_pos_pairs_start_anchor(J, edge_index, num_nodes, walk_length=10, top_k=5, eps=1e-6):
    neighbors_list = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbors_list[u].append(v)
    neighbors_tensor = [torch.tensor(neigh, dtype=torch.long) for neigh in neighbors_list]

    pos_pairs = []

    for start in range(num_nodes):
        cur = start
        visited = {start: 1}

        for step in range(1, walk_length + 1):
            nbrs = neighbors_tensor[cur]
            if nbrs.numel() == 0:
                break
            probs = J[start, nbrs] + eps
            probs /= probs.sum()
            cur = nbrs[torch.multinomial(probs, 1).item()]
            if cur not in visited:
                visited[cur] = step + 1

        borda = {v: visited[v] for v in visited}
        topk = sorted(borda, key=borda.get)[:top_k]
        max_rank = max([borda[v] for v in topk])

        for v in topk:
            if v != start:
                weight = 1.0 - (borda[v] / (max_rank + 1e-6))
                pos_pairs.append((start, v, weight))

    return pos_pairs

def contrastive_loss_weighted_fixed(emb, pos_pairs, temperature=0.5, neg_multiplier=20):
    device = emb.device
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=device)

    pos_i = torch.tensor([p[0] for p in pos_pairs], device=device)
    pos_j = torch.tensor([p[1] for p in pos_pairs], device=device)
    weights = torch.tensor([max(p[2], 0.05) for p in pos_pairs], device=device)

    pos_sim = (emb[pos_i] * emb[pos_j]).sum(dim=1) / temperature
    pos_exp = torch.exp(pos_sim)

    num_nodes = emb.size(0)
    neg_idx = torch.randint(0, num_nodes, (len(pos_i), neg_multiplier), device=device)
    neg_emb = emb[neg_idx]
    pi_emb = emb[pos_i].unsqueeze(1)
    neg_sim = (pi_emb * neg_emb).sum(dim=2) / temperature
    neg_exp_sum = torch.exp(neg_sim).sum(dim=1)

    denom = pos_exp + neg_exp_sum + 1e-8
    loss = -(weights * torch.log(pos_exp / denom)).mean()
    return loss
    
########### - SOME EXPERIMENTS ####################################
# -------------------------------------------------
# POSITIVE HELPERS (unchanged usage)
# -------------------------------------------------
def build_pos_set(pos_pairs):
    return set((i, j) for i, j, _ in pos_pairs)


# -------------------------------------------------
# MEDIUM-HARD NEGATIVE SAMPLING (KEY CHANGE)
# -------------------------------------------------
def sample_mixed_negatives(J, pos_i, num_nodes, topk=20, n_rand=5, n_struct=5):
    """
    Balanced negatives:
    - random negatives (easy)
    - low-similarity structural negatives (medium-hard)
    """

    device = J.device

    # low similarity = hardest safe region (NOT top-k)
    low_sim = torch.topk(-J, k=topk, dim=1).indices

    negs = []

    for i in pos_i:
        i = i.item()

        # -----------------------------
        # 1. random negatives
        # -----------------------------
        rand_negs = torch.randint(
            0, num_nodes, (n_rand,), device=device
        )

        # -----------------------------
        # 2. structural "safe-hard" negatives
        # (low similarity, not top-k)
        # -----------------------------
        struct_pool = low_sim[i]

        if len(struct_pool) >= n_struct:
            struct_negs = struct_pool[:n_struct]
        else:
            struct_negs = torch.randint(
                0, num_nodes, (n_struct,), device=device
            )

        mixed = torch.cat([rand_negs, struct_negs])

        negs.append(mixed[: (n_rand + n_struct)])

    return negs


# -------------------------------------------------
# MAIN LOSS
# -------------------------------------------------
def contrastive_loss_mixed_negatives(
    emb,
    pos_pairs,
    J,
    temperature=0.5
):
    device = emb.device

    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=device)

    # -----------------------------------------
    # positives
    # -----------------------------------------
    pos_i = torch.tensor([p[0] for p in pos_pairs], device=device)
    pos_j = torch.tensor([p[1] for p in pos_pairs], device=device)

    weights = torch.tensor(
        [max(p[2], 0.05) for p in pos_pairs],
        device=device
    )

    # -----------------------------------------
    # positive similarity
    # -----------------------------------------
    pos_sim = (emb[pos_i] * emb[pos_j]).sum(dim=1) / temperature
    pos_exp = torch.exp(pos_sim)

    # -----------------------------------------
    # mixed negatives
    # -----------------------------------------
    negs = sample_mixed_negatives(
        J,
        pos_i,
        emb.size(0),
        topk=20,
        n_rand=5,
        n_struct=5
    )

    neg_exp_list = []

    for idx, i in enumerate(pos_i):

        neg_nodes = negs[idx]

        anchor = emb[i].unsqueeze(0)
        neg_emb = emb[neg_nodes]

        neg_sim = (anchor * neg_emb).sum(dim=1) / temperature

        neg_exp_list.append(torch.exp(neg_sim).sum())

    neg_exp_sum = torch.stack(neg_exp_list)

    # -----------------------------------------
    # InfoNCE
    # -----------------------------------------
    denom = pos_exp + neg_exp_sum + 1e-8

    loss = -(weights * torch.log(pos_exp / denom)).mean()

    return loss