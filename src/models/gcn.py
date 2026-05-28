import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import dense_to_sparse


class GSBlock(nn.Module):
    def __init__(self, feature_dim, embed_dim, policy="mean", gcn=False):
        super().__init__()
        if policy not in {"mean", "max"}:
            raise ValueError("policy must be 'mean' or 'max'")
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.conv = SimpleConv(aggr=policy)
        self.weight = nn.Parameter(
            torch.empty(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        if not torch.is_tensor(adj):
            adj = torch.as_tensor(adj, device=x.device)
        else:
            adj = adj.to(x.device)

        if adj.ndim != 2:
            raise ValueError("adj must be a 2D adjacency matrix")

        if not self.gcn:
            adj = adj.clone()
            adj.fill_diagonal_(0)

        edge_index, _ = dense_to_sparse(adj)
        neigh_feats = self.conv(x, edge_index)

        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)
        return combined
