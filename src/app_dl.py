# src/app_dl.py

import os
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
from pathlib import Path
import uuid
from werkzeug.security import generate_password_hash, check_password_hash

from src.inference_dl import load_resnet, predict_resnet

# ------------------ PATH SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(BASE_DIR, 'templates')
STATIC = os.path.join(BASE_DIR, 'static')
UPLOAD = Path(os.path.join(BASE_DIR, "uploads_dl"))
UPLOAD.mkdir(exist_ok=True)

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "resnet50_checkpoint.pth")
)

# ------------------ FLASK APP ------------------
app = Flask(
    __name__,
    template_folder=TEMPLATES,
    static_folder=STATIC
)
app.secret_key = os.environ.get('FLASK_SECRET', 'change-this-key-in-production')

# ------------------ LOAD MODEL ------------------
try:
    resnet_tuple = load_resnet(MODEL_PATH)
except Exception as e:
    print("ResNet load failed (train first):", e)
    resnet_tuple = None

# ------------------ DATABASE SETUP ------------------
DB_PATH = os.path.join(BASE_DIR, "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ------------------ HELPERS ------------------
def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, email, password_hash FROM users WHERE email=?", (email,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2], "password_hash": row[3]}
    return None

def create_user(name, email, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, generate_password_hash(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# ------------------ ROUTES ------------------

# Landing page -> Login
@app.route("/", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = get_user_by_email(email)

        if not user or not check_password_hash(user['password_hash'], password):
            error = "Invalid email or password"
        else:
            session["user_id"] = user['id']
            session["user_name"] = user['name']
            return redirect(url_for("index"))

    return render_template("login.html", error=error)

# Signup page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            return render_template("signup.html", error="Email & Password required")

        if not create_user(name, email, password):
            return render_template("signup.html", error="Email already exists")

        return redirect(url_for("login"))

    return render_template("signup.html")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# Home page
@app.route("/index")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# About page
@app.route("/about")
def about():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")

# Feedback page
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if "user_id" not in session:
        return redirect(url_for("login"))

    submitted = False
    if request.method == "POST":
        data = request.form.to_dict()
        # Save feedback to CSV for simplicity
        with open(os.path.join(BASE_DIR, "feedback_dl.csv"), "a", encoding="utf-8") as f:
            f.write(",".join([
                data.get("image", ""),
                data.get("predicted", ""),
                data.get("correct", ""),
                data.get("user_label", ""),
                data.get("notes", "")
            ]) + "\n")
        submitted = True

    return render_template("feedback.html", submitted=submitted)

# Serve uploaded files
@app.route("/uploads_dl/<path:filename>")
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD, filename)

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    url_input = request.form.get("image_url", "").strip()

    if file and file.filename:
        fname = UPLOAD / (str(uuid.uuid4()) + "_" + file.filename)
        file.save(fname)
        image_input = str(fname)
        image_url = f"/uploads_dl/{fname.name}"

    elif url_input:
        image_input = url_input
        image_url = url_input

    else:
        return {"error": "No image provided"}, 400

    if resnet_tuple is None:
        return {"error": "Model not available"}, 500

    try:
        label, prob = predict_resnet(image_input, resnet_tuple)
    except Exception as e:
        return {"error": "Prediction failed", "detail": str(e)}, 500

    return {
        "predicted": label,
        "probability": prob,
        "image_path": image_url
    }

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(port=5002, debug=True)
