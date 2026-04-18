# 🎬 GNN-Based Movie Recommendation System

## 📌 Overview

This project is a **Graph Neural Network (GNN) based Movie Recommendation System** built using **Streamlit**.
It recommends movies to users based on **similar user preferences** using graph-based learning.

The system models relationships between:

* 👤 Users
* 🎬 Movies
* ⭐ Ratings

and uses a **Graph Convolutional Network (GCN)** to generate personalized recommendations.

---

## 🚀 Features

### 👥 Guest Mode

* Enter User ID
* Get movie recommendations
* View:

  * Similar users
  * Highlight graph
  * Focus graph
  * Full graph visualization

---

### 🔐 User Authentication

* User Signup with OTP verification (Email)
* User Login with credentials
* Forgot Password with OTP reset
* Session-based login system

---

### 👤 User Mode

* Select favorite movies
* Provide ratings
* Generate personalized recommendations
* View:

  * Similar users
  * Recommendation results
  * Graph visualizations

---

### 🛠️ Admin Panel

* Upload new `ratings.csv`
* Manage dataset
* Preview data

---

## 🧠 How It Works

1. **Graph Construction**

   * Users and Movies are nodes
   * Ratings are weighted edges

2. **GNN Model**

   * Embedding layer
   * Two GCN layers
   * Normalization for similarity

3. **Recommendation Logic**

   * Find similar users using cosine similarity
   * Aggregate ratings
   * Rank movies using:

     ```
     Score = Average Rating × Number of Ratings
     ```

---

## 📊 Graph Visualization

* Built using **PyVis**
* Interactive graph like Neo4j
* Features:

  * Highlight specific user
  * Show user-movie connections
  * Show similar user network

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Model:** PyTorch + PyTorch Geometric
* **Graph Visualization:** PyVis
* **Data Handling:** Pandas
* **Email System:** SMTP (Gmail)

---

## 📁 Project Structure

```
project/
│
├── app.py
├── ratings.csv
├── movies.csv
├── users.csv
├── README.md
```

---

## ⚙️ Installation

### 1. Install Dependencies

```
pip install streamlit pandas torch torch-geometric pyvis
```

---

## ▶️ Run the App

```
streamlit run app.py
```

---

## 📧 Email Setup (IMPORTANT)

To enable OTP system:

1. Enable **2-Step Verification** in Gmail
2. Generate **App Password**
3. Replace in code:

```
sender_email = "your_email@gmail.com"
app_password = "your_16_char_app_password"
```

---

## 🔐 Authentication Flow

### Signup

* Enter email + password
* Receive OTP via email
* Verify → Account created

### Login

* Enter credentials
* Access user features

### Forgot Password

* Enter email
* Receive OTP
* Reset password

---

## ⚠️ Known Limitations

* Password stored in plain text (for academic use only)
* Small dataset may limit recommendation diversity
* Requires manual retraining after new user addition

---

## 🔮 Future Improvements

* 🔐 Password hashing (security)
* 📊 Larger dataset integration
* ⚡ Real-time model retraining
* 🌐 Deployment (Streamlit Cloud / AWS)
* 🤖 Hybrid recommendation (Content + Collaborative)

---

## 🎯 Use Cases

* Movie streaming platforms
* Recommendation systems research
* Graph-based ML learning
* Academic projects

---

## 👨‍💻 Author

* S.S.SHRIRAM
* Pre-Final Year Recommendation System Mini-Project

---

## ⭐ Acknowledgment

This project demonstrates the use of **Graph Neural Networks** for real-world recommendation systems.

---

## 📌 Note

This project is developed for **educational purposes** and demonstrates:

* GNN concepts
* Recommendation systems
* Full-stack ML app integration

---
