import torch

RCOV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
VALENCE = {1: 1,    6: 4,    7: 3,    8: 2}

# Pairwise slack and caps (Å)
PAIR_S = {
    (1, 6): 1.2, (1, 7): 1.2, (1, 8): 1.2,   # X–H stricter
    (6, 6): 1.2, (6, 7): 1.2, (6, 8): 1.2, (7, 7): 1.2, (7, 8): 1.2, (8, 8): 1.2,
}
PAIR_CAP = {
    (1, 6): 1.50, (1, 7): 1.50, (1, 8): 1.50,
    (6, 6): 1.75, (6, 7): 1.65, (6, 8): 1.65, (7, 7): 1.65, (7, 8): 1.60, (8, 8): 1.55,
}

def _pairkey(a, b):
    a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)

def build_bonds_pairwise_torch(positions_A: torch.Tensor,
                               Z: torch.Tensor,
                               dmin: float = 0.60,
                               base_s: float = 1.15) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Two-stage: heavy–heavy first, then assign hydrogens.
    Returns edges (E,2) and lengths (E,)
    """
    device = positions_A.device
    dtype = positions_A.dtype
    N = positions_A.shape[0]

    r = torch.tensor([RCOV.get(int(z), 0.77) for z in Z.tolist()], dtype=dtype, device=device)
    vcap = torch.tensor([VALENCE.get(int(z), 4) for z in Z.tolist()], dtype=torch.long, device=device)

    D = torch.cdist(positions_A, positions_A)
    iu = torch.triu_indices(N, N, offset=1, device=device)
    i_idx, j_idx = iu[0], iu[1]
    dij = D[i_idx, j_idx]

    # --- Stage 1: heavy–heavy only (forbid H–H) ---
    is_H_i = (Z[i_idx] == 1)
    is_H_j = (Z[j_idx] == 1)
    hh_mask = ~(is_H_i | is_H_j)  # both not H

    # pairwise s and caps
    s_pair = torch.full_like(dij, base_s)
    cap_pair = torch.full_like(dij, float('inf'))
    for k in range(i_idx.numel()):
        Zi, Zj = int(Z[i_idx[k]]), int(Z[j_idx[k]])
        key = _pairkey(Zi, Zj)
        if key in PAIR_S:
            s_pair[k] = PAIR_S[key]
        if key in PAIR_CAP:
            cap_pair[k] = PAIR_CAP[key]

    sum_r = r[i_idx] + r[j_idx]
    keep_hh = hh_mask & (dij >= dmin) & (dij <= s_pair * sum_r) & (dij <= cap_pair)

    edges_hh = torch.stack([i_idx[keep_hh], j_idx[keep_hh]], dim=1)
    lengths_hh = dij[keep_hh]

    # Greedy valence prune for heavy–heavy (shorter first)
    order = torch.argsort(lengths_hh)
    degree = torch.zeros(N, dtype=torch.long, device=device)
    kept = torch.zeros_like(order, dtype=torch.bool)
    for t, k in enumerate(order.tolist()):
        i, j = edges_hh[k]
        if degree[i] < vcap[i] and degree[j] < vcap[j]:
            kept[t] = True
            degree[i] += 1
            degree[j] += 1
    edges_hh = edges_hh[kept]
    lengths_hh = lengths_hh[kept]

    # --- Stage 2: assign each H to one nearest heavy with remaining capacity ---
    H_idx = torch.nonzero(Z == 1, as_tuple=False).flatten()
    heavy_idx = torch.nonzero(Z != 1, as_tuple=False).flatten()

    edges_list = [edges_hh]
    lens_list = [lengths_hh]

    if H_idx.numel() > 0 and heavy_idx.numel() > 0:
        # distances H to heavy
        D_HH = torch.cdist(positions_A[H_idx], positions_A[heavy_idx])
        # try up to k nearest heavy atoms per H
        k = min(3, D_HH.shape[1])
        nnk = torch.topk(D_HH, k=k, largest=False, dim=1).indices  # (nH, k)
        cap_relax = 0.02  # Å small fallback if strict cap fails
        for hpos, h_atom in enumerate(H_idx.tolist()):
            attached = False
            for kk in range(k):
                hvy = heavy_idx[nnk[hpos, kk]].item()
                d = float(D_HH[hpos, nnk[hpos, kk]])
                key = _pairkey(1, int(Z[hvy]))
                s_h = PAIR_S.get(key, 1.12)
                cap_h = PAIR_CAP.get(key, 1.30)
                # strict check
                if d >= dmin and d <= s_h * (RCOV[1] + RCOV.get(int(Z[hvy]), 0.77)) and d <= cap_h:
                    if degree[h_atom] < vcap[h_atom] and degree[hvy] < vcap[hvy]:
                        edges_list.append(torch.tensor([[h_atom, hvy]], device=device))
                        lens_list.append(torch.tensor([d], dtype=dtype, device=device))
                        degree[h_atom] += 1
                        degree[hvy] += 1
                        attached = True
                        break
            if not attached:
                # tiny relaxed cap attempt with the nearest heavy only
                hvy = heavy_idx[nnk[hpos, 0]].item()
                d = float(D_HH[hpos, nnk[hpos, 0]])
                key = _pairkey(1, int(Z[hvy]))
                s_h = PAIR_S.get(key, 1.12)
                cap_h = PAIR_CAP.get(key, 1.30) + cap_relax
                if d >= dmin and d <= s_h * (RCOV[1] + RCOV.get(int(Z[hvy]), 0.77)) and d <= cap_h:
                    if degree[h_atom] < vcap[h_atom] and degree[hvy] < vcap[hvy]:
                        edges_list.append(torch.tensor([[h_atom, hvy]], device=device))
                        lens_list.append(torch.tensor([d], dtype=dtype, device=device))
                        degree[h_atom] += 1
                        degree[hvy] += 1
        # (implicitly forbids H–H and multiple bonds from one H)

    edges = torch.vstack(edges_list) if len(edges_list) > 0 else torch.empty((0,2), dtype=torch.long, device=device)
    lengths = torch.cat(lens_list) if len(lens_list) > 0 else torch.empty((0,), dtype=dtype, device=device)
    return edges, lengths