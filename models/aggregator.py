import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAggregator(nn.Module):

    def __init__(self, st_dim, proj_dim):
        super().__init__()
        self.W1 = nn.Linear(st_dim, proj_dim, bias=False)
        self.W2 = nn.Linear(st_dim, proj_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(proj_dim))
        self.W_back = nn.Linear(proj_dim, st_dim)


def build_adj(n, neighbors_dict, device):
    rows, cols = [], []
    for i in range(n):
        nbrs = neighbors_dict.get(i, set())
        for j in nbrs:
            rows.append(i)
            cols.append(j)
    if len(rows) == 0:
        return torch.zeros(n, n, device=device)
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    vals = torch.ones(len(rows), device=device)
    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (n, n)
    ).to_dense()
    return A


def aggregate_meta_path(works, st_embeddings, neighbors_dict, aggregator, L=2):
    n = len(works)
    device = st_embeddings.device

    A = build_adj(n, neighbors_dict, device)  # (n, n)

    center = st_embeddings
    order_embeddings = []

    for _ in range(L):
        neighbor_sum = A @ st_embeddings        # (n, st_dim)
        w1_center = aggregator.W1(center)       # (n, proj_dim)
        w1_neigh  = aggregator.W1(neighbor_sum) # (n, proj_dim)
        w2_neigh  = aggregator.W2(neighbor_sum * center) # (n, proj_dim)
        out = F.relu(w1_center + w1_neigh + w2_neigh + aggregator.bias)
        order_embeddings.append(out)
        center = torch.tanh(aggregator.W_back(out))

    return torch.cat(order_embeddings, dim=1)
