from flask import Flask, render_template, request, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import random
import os
from recommend import get_recommendations

app = Flask(__name__)
app.secret_key = "secret"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(20), default="user")

otp_store = {}

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['role'] = user.role
            return redirect("/dashboard")

    return render_template("login.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        otp = str(random.randint(1000,9999))
        otp_store[email] = (otp, password)

        print("OTP:", otp)

        return render_template("verify.html", email=email)

    return render_template("signup.html")

@app.route("/verify", methods=["POST"])
def verify():
    email = request.form['email']
    otp = request.form['otp']

    if email in otp_store and otp_store[email][0] == otp:
        password = otp_store[email][1]

        user = User(email=email, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()

        return redirect("/")

    return "Invalid OTP"

@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

@app.route("/guest")
def guest():
    session['user_id'] = 1
    session['role'] = "guest"
    return redirect("/dashboard")

@app.route("/recommend")
def recommend():
    return jsonify(get_recommendations(session.get("user_id",1)))

@app.route("/search")
def search():
    q = request.args.get("q")
    movies = pd.read_csv("data/movies.csv")
    result = movies[movies['title'].str.contains(q, case=False)]
    return jsonify(result.to_dict("records"))

@app.route("/admin")
def admin():
    if session.get("role") != "admin":
        return "Access Denied"

    ratings = pd.read_csv("data/ratings.csv")

    return render_template("admin.html",
                           users=ratings['userId'].nunique(),
                           movies=ratings['movieId'].nunique(),
                           ratings=len(ratings))

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['file']
    file.save("data/ratings.csv")
    return "Uploaded"

@app.route("/retrain")
def retrain():
    os.system("python train.py")
    return "Model Retrained"

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)