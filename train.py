import torch
import pandas as pd
import random
from torch_geometric.data import Data
from gnn_model import LightGCN

ratings = pd.read_csv("data/ratings.csv")

users = ratings['userId'].unique()
movies = ratings['movieId'].unique()

u_map = {u:i for i,u in enumerate(users)}
m_map = {m:i for i,m in enumerate(movies)}

ratings['u'] = ratings['userId'].map(u_map)
ratings['i'] = ratings['movieId'].map(m_map)

num_u = len(u_map)
num_m = len(m_map)

# Graph
edges = []
for _, r in ratings.iterrows():
    edges.append([r['u'], r['i'] + num_u])
    edges.append([r['i'] + num_u, r['u']])

edge_index = torch.tensor(edges).t().contiguous()
data = Data(edge_index=edge_index)

# Positive interactions
user_pos = {}
for _, r in ratings.iterrows():
    user_pos.setdefault(r['u'], set()).add(r['i'])

model = LightGCN(num_u, num_m)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def bpr_loss(u, i, j):
    pos = (u * i).sum(1)
    neg = (u * j).sum(1)
    return -torch.mean(torch.log(torch.sigmoid(pos - neg)))

for epoch in range(20):
    user_emb, item_emb = model(data.edge_index)

    users_list, pos_list, neg_list = [], [], []

    for u in user_pos:
        for p in user_pos[u]:
            n = random.randint(0, num_m-1)
            while n in user_pos[u]:
                n = random.randint(0, num_m-1)

            users_list.append(u)
            pos_list.append(p)
            neg_list.append(n)

    u_t = torch.tensor(users_list)
    p_t = torch.tensor(pos_list)
    n_t = torch.tensor(neg_list)

    loss = bpr_loss(user_emb[u_t], item_emb[p_t], item_emb[n_t])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save({
    "model": model.state_dict(),
    "u_map": u_map,
    "m_map": m_map
}, "model.pth")

print("✅ Model trained successfully")