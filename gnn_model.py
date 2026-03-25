import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, dim=64, layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

        self.convs = nn.ModuleList([LGConv() for _ in range(layers)])

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight])
        all_emb = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_emb.append(x)

        x = torch.stack(all_emb).mean(0)
        return torch.split(x, [self.num_users, self.num_items])