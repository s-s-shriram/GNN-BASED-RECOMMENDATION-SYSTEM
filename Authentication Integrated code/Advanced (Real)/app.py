# It generate fixed OTP on same screen and collect SignUp credential and used for log in
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

#--------------EMAIL FUNCTION --------------------
import smtplib
from email.mime.text import MIMEText

def send_email_otp(receiver_email, otp, purpose="verify", username="User"):
    
    sender_email = "ssshriram1982@gmail.com"
    app_password = "hxhf ztjv mzdn tjzs"

    # 🔗 Fake verification link (you can later connect to backend)
    verify_link = f"http://localhost:8501/?verify_email={receiver_email}&otp={otp}"

    # 🎯 Different subject/content
    if purpose == "reset":
        subject = "🔐 Reset Your Password | SSS MOVIE RS"
        title = "Password Reset Request"
        message = "You requested to reset your password."
    else:
        subject = "✅ Verify Your Email | SSS MOVIE RS"
        title = "Welcome to SSS MOVIE RS 🎬"
        message = "Thank you for signing up with us."

    body = f"""
    <html>
    <body style="font-family: Arial; background-color:#f4f4f4; padding:20px;">
    
    <div style="max-width:600px; margin:auto; background:white; padding:20px; border-radius:10px;">
    
        <h2 style="color:#e50914; text-align:center;">🎬 SSS MOVIE RS</h2>
        
        <p>Hello <b>{username}</b>,</p>

        <p>{message}</p>

        <p style="font-size:16px;">Use the OTP below:</p>

        <h1 style="color:#ff4b4b; text-align:center;">{otp}</h1>

        <p style="text-align:center;">OR</p>

        <div style="text-align:center; margin:20px;">
            <a href="{verify_link}" 
               style="background:#4CAF50; color:white; padding:12px 20px; 
                      text-decoration:none; border-radius:5px;">
               ✅ Verify Email
            </a>
        </div>

        <p style="font-size:14px;">
        This OTP is valid for a short time. Do not share it.
        </p>

        <hr>

        <p style="font-size:12px; color:gray;">
        This is an automated email. Please do not reply.
        </p>

        <p>
        Regards,<br>
        <b>S.S.SHRIRAM<b><br>
        <b>SSS MOVIE RS Team</b>
        </p>

    </div>
    </body>
    </html>
    """

    msg = MIMEText(body, "html")
    msg["Subject"] = subject
    msg["From"] =f"SSS Movie Recommender Team <{sender_email}>"
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(e)
        return False
    
# ---------------- LOAD DATA ----------------
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# ---------------- BUILD GRAPH ----------------
def build_graph(ratings):
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    user_map = {int(u): i for i, u in enumerate(user_ids)}
    movie_map = {int(m): i + len(user_ids) for i, m in enumerate(movie_ids)}

    edges, weights = [], []

    for _, row in ratings.iterrows():
        u = user_map[int(row['userId'])]
        m = movie_map[int(row['movieId'])]
        r = row['rating']

        edges.append([u, m])
        edges.append([m, u])
        weights.append(r)
        weights.append(r)

    return (
        torch.tensor(edges).t().contiguous(),
        torch.tensor(weights, dtype=torch.float),
        user_map,
        movie_map,
        len(user_ids) + len(movie_ids)
    )

edge_index, edge_weight, user_map, movie_map, num_nodes = build_graph(ratings)

# ---------------- MODEL ----------------
class GNNRecommender(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, 64)
        self.conv1 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 32)

    def forward(self, edge_index, edge_weight):
        x = self.embedding.weight
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)

        # 🔥 NORMALIZATION (IMPORTANT)
        x = F.normalize(x, p=2, dim=1)
        return x

model = GNNRecommender(num_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------------- TRAIN ----------------
def train():
    model.train()
    optimizer.zero_grad()

    emb = model(edge_index, edge_weight)
    src, dst = edge_index

    pos = (emb[src] * emb[dst]).sum(dim=1)
    neg_dst = torch.randint(0, num_nodes, dst.size())
    neg = (emb[src] * emb[neg_dst]).sum(dim=1)

    loss = -torch.log(torch.sigmoid(pos)).mean() - torch.log(1 - torch.sigmoid(neg)).mean()

    loss.backward()
    optimizer.step()

# 🔥 TRAIN LONGER (IMPORTANT)
for _ in range(100):
    train()

# ---------------- SIMILAR USERS ----------------
def get_similar_users(user_id, emb, user_map):
    user_id = int(user_id)
    user_idx = user_map[user_id]
    user_emb = emb[user_idx]

    sims = []

    for u, idx in user_map.items():
        if u != user_id:
            sim = F.cosine_similarity(
                user_emb.unsqueeze(0),
                emb[idx].unsqueeze(0)
            ).item()

            sims.append((int(u), sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    return [u for u, _ in sims[:5]]

# ---------------- RECOMMEND ----------------
def recommend(user_id):
    user_id = int(user_id)

    model.eval()
    with torch.no_grad():
        emb = model(edge_index, edge_weight)

    sim_users = get_similar_users(user_id, emb, user_map)

    sim_data = ratings[ratings['userId'].isin(sim_users)]
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    # 🔥 REMOVE WATCHED MOVIES
    rec = sim_data[~sim_data['movieId'].isin(user_movies)]

    # 🔥 AGGREGATE (IMPORTANT FIX)
    rec_grouped = rec.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()

    rec_grouped.columns = ['movieId', 'avg_rating', 'count']

    # 🔥 SCORE = rating + popularity
    rec_grouped['score'] = rec_grouped['avg_rating'] * rec_grouped['count']

    # 🔥 SORT
    rec_grouped = rec_grouped.sort_values(by='score', ascending=False)

    # 🔥 GET MORE MOVIES (10 instead of 5)
    top_movies = rec_grouped['movieId'].head(10)

    return movies[movies['movieId'].isin(top_movies)][['title']], sim_users

# ---------------- RECOMMEND ----------------
def personalized_recommend(user_id, model, edge_index, edge_weight, user_map):
    model.eval()
    with torch.no_grad():
        emb = model(edge_index, edge_weight)

    sim_users = get_similar_users(user_id, emb, user_map)

    sim_data = ratings[ratings['userId'].isin(sim_users)]
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    rec = sim_data[~sim_data['movieId'].isin(user_movies)]

    # 🔥 AGGREGATE
    rec_grouped = rec.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()

    rec_grouped.columns = ['movieId', 'avg_rating', 'count']

    rec_grouped['score'] = rec_grouped['avg_rating'] * rec_grouped['count']

    rec_grouped = rec_grouped.sort_values(by='score', ascending=False)

    top_movies = rec_grouped['movieId'].head(10)

    return movies[movies['movieId'].isin(top_movies)][['title']], sim_users

# ---------------- ADD NEW USER ----------------
def add_user(selected_movies, ratings_input):
    global ratings

    new_user_id = ratings['userId'].max() + 1

    new_data = []
    for m, r in zip(selected_movies, ratings_input):
        new_data.append({
            "userId": new_user_id,
            "movieId": m,
            "rating": r
        })

    ratings = pd.concat([ratings, pd.DataFrame(new_data)], ignore_index=True)
    ratings.to_csv("ratings.csv", index=False)

    return new_user_id

# ---------------- GRAPH ----------------
def show_graph(user_id=None, sim_users=None):
    net = Network(height="650px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut()

    if user_id:
        user_id = int(user_id)
    if sim_users:
        sim_users = [int(u) for u in sim_users]

    if user_id:
        focus = ratings[ratings['userId'] == user_id]
        sample = pd.concat([ratings.sample(200), focus])
    else:
        sample = ratings.sample(300)

    for _, row in sample.iterrows():
        uid = int(row['userId'])
        mid = int(row['movieId'])

        user_node = f"User {uid}"
        movie_node = f"Movie {mid}"

        # Color logic
        if uid == user_id:
            color = "yellow"
            size = 25
        elif sim_users and uid in sim_users:
            color = "green"
            size = 18
        else:
            color = "red"
            size = 10

        net.add_node(user_node, color=color, size=size)
        net.add_node(movie_node, color="blue", size=8)

        if row['rating'] >= 3:
            net.add_edge(user_node, movie_node, title=f"Rating: {row['rating']}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name).read()
    components.html(html, height=650)

# ---------------- FOCUS GRAPH ----------------
def show_focus(user_id):
    user_id = int(user_id)

    net = Network(height="650px", width="100%", bgcolor="black", font_color="white")

    user_data = ratings[ratings['userId'] == user_id]

    net.add_node(f"User {user_id}", color="yellow", size=30)

    for _, row in user_data.iterrows():
        movie = f"Movie {int(row['movieId'])}"
        net.add_node(movie, color="blue")
        net.add_edge(f"User {user_id}", movie, title=f"Rating: {row['rating']}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name).read()
    components.html(html, height=650)
    
# ---------------- FOCUSED GRAPH ----------------
def show_focus_graph(user_id):
    user_id = int(user_id)

    net = Network(height="650px", width="100%", bgcolor="#000", font_color="white")

    net.barnes_hut()

    user_data = ratings[ratings['userId'] == user_id]

    # Add main user
    net.add_node(f"User {user_id}", color="yellow", size=30)

    for _, row in user_data.iterrows():
        mid = int(row['movieId'])
        movie_node = f"Movie {mid}"

        net.add_node(movie_node, color="blue", size=15)

        net.add_edge(
            f"User {user_id}",
            movie_node,
            title=f"Rating: {row['rating']}"
        )

    net.set_options("""
    var options = {
      "interaction": {
        "hover": true
      }
    }
    """)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name, "r", encoding="utf-8").read()
    components.html(html, height=650)

# ---------------- USER SIM GRAPH ----------------
def show_user_sim_graph(user_id):
    net = Network(height="650px", width="100%", bgcolor="black", font_color="white")

    model.eval()
    with torch.no_grad():
        emb = model(edge_index, edge_weight)

    user_id = int(user_id)
    user_idx = user_map[user_id]
    target = emb[user_idx]

    sims = []
    for u, idx in user_map.items():
        if u != user_id:
            sim = F.cosine_similarity(
                target.unsqueeze(0),
                emb[idx].unsqueeze(0)
            ).item()
            sims.append((int(u), sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:5]

    net.add_node(f"User {user_id}", color="yellow", size=30)

    for u, sim in sims:
        net.add_node(f"User {u}", color="green")
        net.add_edge(f"User {user_id}", f"User {u}", title=f"{sim:.3f}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name).read()
    components.html(html, height=650)
    
# ---------------- AUTH SYSTEM ----------------
import random

USER_FILE = "users.csv"

def load_users():
    try:
        df = pd.read_csv(USER_FILE)

        # ✅ Ensure required columns exist
        required_cols = ["name", "email", "password"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        return df

    except:
        # ✅ Create proper structure if file doesn't exist
        return pd.DataFrame(columns=["name", "email", "password"])

def save_user(name, email, password):
    df = load_users()

    new_user = pd.DataFrame([{
        "name": name,
        "email": email,
        "password": password
    }])

    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_FILE, index=False)

def authenticate(email, password):
    df = load_users()
    user = df[(df['email'] == email) & (df['password'] == password)]

    if not user.empty:
        return True, user.iloc[0]['name']
    return False, None

def user_exists(email):
    df = load_users()
    return email in df['email'].values

def generate_otp():
    return str(random.randint(1000, 9999))

# ---------------- UI ----------------
# ---------------- UI ----------------
st.title("🎬 SSSHRIRAM GNN - Movie Recommender System ")

if "auth" not in st.session_state:
    st.session_state.auth = None

menu = st.sidebar.selectbox("Select Role", ["Guest", "User Login", "User Signup", "Admin Login"])

# ---------------- GUEST ----------------
if menu == "Guest":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if "recs" not in st.session_state:
        st.session_state.recs = None
    if "sim" not in st.session_state:
        st.session_state.sim = None

    if st.button("Get Recommendations"):
        recs, sim = recommend(user_id)
        st.session_state.recs = recs
        st.session_state.sim = sim

    if st.session_state.recs is not None:
        st.write("### Similar Users")
        st.write(st.session_state.sim)

        st.write("### Recommendations")
        st.table(st.session_state.recs)

        if st.button("🔍 Show Highlight Graph", key="guest_highlight"):
            show_graph(user_id, st.session_state.sim)

        if st.button("🎯 Show Focus Graph", key="guest_focus"):
            show_focus(user_id)

        if st.button("👥 Show User Similarity Graph", key="guest_sim"):
            show_user_sim_graph(user_id)

        if st.button("📊 Show Full Graph", key="guest_full"):
            show_graph()

# ---------------- USER SIGNUP ----------------
elif menu == "User Signup":
    st.header("📝 Signup")
    name = st.text_input("Full Name")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    # INIT SESSION STATE
    if "signup_otp" not in st.session_state:
        st.session_state.signup_otp = None

    if "temp_user" not in st.session_state:
        st.session_state.temp_user = None

    # STEP 1: REGISTER → SEND OTP
    if st.button("Register"):
        if user_exists(email):
            st.error("User already exists")
        else:
            otp = generate_otp()
            st.session_state.signup_otp = otp
            st.session_state.temp_user = (name, email, password)

            if send_email_otp(email, otp, name):
                st.success("OTP sent to your email")
            else:
                st.error("Failed to send email")

    # STEP 2: SHOW OTP INPUT ONLY AFTER REGISTER
    if st.session_state.signup_otp:
        otp_input = st.text_input("Enter OTP")

        if st.button("Verify OTP"):
            if otp_input == st.session_state.signup_otp:
                name, email, password = st.session_state.temp_user
                save_user(name, email, password)

                st.success("Signup successful 🎉")

                # RESET STATE
                st.session_state.signup_otp = None
                st.session_state.temp_user = None
            else:
                st.error("Invalid OTP")

# ---------------- USER LOGIN ----------------
# ---------------- USER LOGIN ----------------
elif menu == "User Login":
    st.header("🔐 Login")

    # ---------------- SESSION DEFAULTS ----------------
    if "reset_otp" not in st.session_state:
        st.session_state.reset_otp = None

    if "reset_email" not in st.session_state:
        st.session_state.reset_email = None

    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False

    if "show_graph" not in st.session_state:
        st.session_state.show_graph = False

    if "show_focus" not in st.session_state:
        st.session_state.show_focus = False

    # ---------------- LOGIN FORM ----------------
    if st.session_state.auth is None:

        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            status, name = authenticate(email, password)

            if status:
                st.session_state.auth = email
                st.session_state.username = name
                st.success(f"Welcome {name} 👋")
                st.rerun()   # 🔥 IMPORTANT FIX
            else:
                st.error("Invalid credentials")

        # ---------------- FORGOT PASSWORD ----------------
        if st.button("Forgot Password"):
            if user_exists(email):
                otp = generate_otp()
                st.session_state.reset_otp = otp
                st.session_state.reset_email = email

                df = load_users()
                name = df[df['email'] == email]['name'].values[0]

                if send_email_otp(email, otp, name, purpose="reset"):
                    st.success("OTP sent to your email")
                else:
                    st.error("Email sending failed")
            else:
                st.error("Email not found")

        # STEP 2: VERIFY OTP
        if st.session_state.reset_otp:
            entered_otp = st.text_input("Enter OTP")

            if st.button("Verify OTP"):
                if entered_otp == st.session_state.reset_otp:
                    st.session_state.otp_verified = True
                    st.success("OTP Verified ✅")
                else:
                    st.error("Invalid OTP")

        # STEP 3: RESET PASSWORD
        if st.session_state.otp_verified:
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")

            if st.button("Update Password"):
                if new_pass == confirm_pass:
                    df = load_users()
                    df.loc[df['email'] == st.session_state.reset_email, 'password'] = new_pass
                    df.to_csv(USER_FILE, index=False)

                    st.success("Password updated successfully 🎉")

                    # RESET SESSION
                    st.session_state.reset_otp = None
                    st.session_state.otp_verified = False
                else:
                    st.error("Passwords do not match")

    # ---------------- AFTER LOGIN ----------------
    else:
        st.success(f"👤 Logged in as: {st.session_state.username}")

        # LOGOUT
        if st.button("Logout"):
            st.session_state.auth = None
            st.session_state.username = None
            st.session_state.show_graph = False
            st.session_state.show_focus = False
            st.rerun()

        st.header("🎯 Get Recommendations")

        movie_options = movies[['movieId', 'title']]

        selected_movies = st.multiselect(
            "Select movies you like",
            options=movie_options['movieId'],
            format_func=lambda x: movie_options[movie_options['movieId']==x]['title'].values[0]
        )

        ratings_input = []
        for m in selected_movies:
            r = st.slider(
                f"Rate {movies[movies['movieId']==m]['title'].values[0]}",
                1, 5, 3
            )
            ratings_input.append(r)

        # ---------------- GENERATE RECOMMENDATIONS ----------------
        if st.button("Submit Preferences"):

            new_user = add_user(selected_movies, ratings_input)

            updated = pd.read_csv("ratings.csv")
            ei, ew, um, mm, nnodes = build_graph(updated)

            new_model = GNNRecommender(nnodes)
            opt = torch.optim.Adam(new_model.parameters(), lr=0.01)

            for _ in range(50):
                new_model.train()
                opt.zero_grad()

                emb = new_model(ei, ew)
                src, dst = ei

                pos = (emb[src] * emb[dst]).sum(dim=1)
                neg_dst = torch.randint(0, nnodes, dst.size())
                neg = (emb[src] * emb[neg_dst]).sum(dim=1)

                loss = -torch.log(torch.sigmoid(pos)).mean() - torch.log(1 - torch.sigmoid(neg)).mean()

                loss.backward()
                opt.step()

            recs, sim_users = personalized_recommend(new_user, new_model, ei, ew, um)

            # SAVE TO SESSION
            st.session_state.user_id = new_user
            st.session_state.sim_users = sim_users
            st.session_state.recs = recs

            st.success(f"🎉 Welcome {st.session_state.username}")
            st.success(f"Your User ID: {new_user}")

        # ---------------- SHOW RESULTS (PERSISTENT) ----------------
        if "recs" in st.session_state and st.session_state.recs is not None:

            st.subheader("👥 Similar Users")
            st.write(st.session_state.sim_users)

            st.subheader("🎬 Recommendations")
            st.table(st.session_state.recs)

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔍 Show My Graph"):
                    st.session_state.show_graph = True
                    st.session_state.show_focus = False

            with col2:
                if st.button("🎯 My Focus Graph"):
                    st.session_state.show_focus = True
                    st.session_state.show_graph = False

        # ---------------- SHOW GRAPHS (FIXED) ----------------
        if st.session_state.show_graph:
            st.subheader("🔍 User Similarity Graph")
            show_graph(
                st.session_state.user_id,
                st.session_state.sim_users
            )

        if st.session_state.show_focus:
            st.subheader("🎯 Focus Graph")
            show_focus_graph(
                st.session_state.user_id
            )
# ---------------- ADMIN ----------------
# ---------------- ADMIN ----------------
elif menu == "Admin Login":
    st.header("🛠️ Admin Login")

    # SESSION DEFAULT
    if "admin" not in st.session_state:
        st.session_state.admin = False

    # ---------------- LOGIN FORM ----------------
    if not st.session_state.admin:

        username = st.text_input("Username", key="admin_user")
        password = st.text_input("Password", type="password", key="admin_pass")

        if st.button("Login Admin"):
            if username == "SSSHRI2058" and password == "SSSadmin2005":
                st.session_state.admin = True
                st.success("Admin Logged In")
            else:
                st.error("Wrong credentials")

    # ---------------- AFTER LOGIN ----------------
    else:
        st.success("🛠️ Logged in as Admin")

        # ✅ LOGOUT BUTTON
        if st.button("Logout Admin"):
            st.session_state.admin = False
            st.rerun()

        st.header("🛠️ Admin Dashboard")

        # ---------------- FILE UPLOAD ----------------
        uploaded = st.file_uploader("Upload New ratings.csv", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)
            df.to_csv("ratings.csv", index=False)
            st.success("Dataset updated")
        

        # ---------------- BASIC METRICS ----------------
        st.subheader("📊 System Overview")

        total_users = ratings['userId'].nunique()
        total_movies = ratings['movieId'].nunique()
        total_ratings = len(ratings)

        col1, col2, col3 = st.columns(3)

        col1.metric("👤 Total Users", total_users)
        col2.metric("🎬 Total Movies", total_movies)
        col3.metric("⭐ Total Ratings", total_ratings)

        # ---------------- MOST ACTIVE USERS ----------------
        st.subheader("🔥 Most Active Users")

        user_activity = ratings.groupby('userId').size().reset_index(name='ratings_count')
        top_users = user_activity.sort_values(by='ratings_count', ascending=False).head(10)

        st.bar_chart(top_users.set_index('userId'))

        # ---------------- TOP MOVIES ----------------
        st.subheader("🎯 Top Rated Movies")

        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()

        movie_stats.columns = ['movieId', 'avg_rating', 'count']

        # Score = rating * popularity
        movie_stats['score'] = movie_stats['avg_rating'] * movie_stats['count']

        top_movies = movie_stats.sort_values(by='score', ascending=False).head(10)

        # Merge with movie titles
        top_movies = top_movies.merge(movies, on='movieId')

        st.bar_chart(top_movies.set_index('title')['score'])

        # ---------------- DATA PREVIEW ----------------
        st.subheader("📄 Raw Data Preview")
        st.dataframe(ratings.head())    
        # ---------------- DOWNLOAD USERS CSV ----------------
        st.subheader("📥 Download Users Data")

        try:
            with open("users.csv", "rb") as file:
                st.download_button(
                    label="⬇️ Download users.csv",
                    data=file,
                    file_name="users.csv",
                    mime="text/csv"
                )
        except FileNotFoundError:
            st.error("users.csv not found")
        
