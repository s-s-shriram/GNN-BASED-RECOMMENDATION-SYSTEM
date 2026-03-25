import torch
import pandas as pd
import requests
from gnn_model import LightGCN

API_KEY = "YOUR_TMDB_API_KEY"

def get_poster(name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={name}"
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500" + data['results'][0]['poster_path']
    except:
        return "https://via.placeholder.com/200x300"

def get_recommendations(user_id):
    data = torch.load("model.pth")

    u_map = data["u_map"]
    m_map = data["m_map"]
    inv_map = {v:k for k,v in m_map.items()}

    model = LightGCN(len(u_map), len(m_map))
    model.load_state_dict(data["model"])
    model.eval()

    ratings = pd.read_csv("data/ratings.csv")

    edges = []
    for _, r in ratings.iterrows():
        edges.append([u_map[r['userId']], m_map[r['movieId']] + len(u_map)])
        edges.append([m_map[r['movieId']] + len(u_map), u_map[r['userId']]])

    edge_index = torch.tensor(edges).t().contiguous()

    user_emb, item_emb = model(edge_index)

    if user_id not in u_map:
        return []

    uid = u_map[user_id]
    scores = torch.matmul(item_emb, user_emb[uid])

    top_items = torch.topk(scores, 6).indices.tolist()

    movies = pd.read_csv("data/movies.csv")

    results = []
    for i in top_items:
        mid = inv_map[i]
        name = movies[movies['movieId']==mid]['title'].values[0]

        results.append({
            "title": name,
            "poster": get_poster(name),
            "id": int(mid)
        })

    return results