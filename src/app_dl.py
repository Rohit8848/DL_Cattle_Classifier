# src/app_dl.py

import os
import secrets
import smtplib
import sqlite3
import uuid
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

from src.breed_info import BREED_INFO
from src.inference_dl import load_efficientnet, predict_efficientnet
from src.detect import detect_cattle


# ------------------ PATH SETUP ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(BASE_DIR, 'templates')
STATIC = os.path.join(BASE_DIR, 'static')
UPLOAD = Path(os.path.join(BASE_DIR, "uploads_dl"))
UPLOAD.mkdir(exist_ok=True)

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "models", "efficientnet_b4.pth")
)

# ------------------ FLASK APP ------------------

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC)
app.secret_key = os.environ.get('FLASK_SECRET', 'change-this-key-in-production')

# ------------------ EMAIL CONFIG ------------------
SENDER_EMAIL = os.environ.get("EMAIL_USER", "sycx63144@gmail.com")
SENDER_PASSWORD = os.environ.get("EMAIL_PASS", "iepmbriyisvcclyj")

# ------------------ BASE URL CONFIG ------------------
# AUTO-DETECTS the correct host from each request — no setup needed.
# The reset link will always use the same host the user is accessing
# the app from (localhost, LAN IP, ngrok URL, or deployed domain).
#
# Only set BASE_URL manually if you're behind a reverse proxy that
# hides the real host:
#   Windows:   set BASE_URL=https://abc123.ngrok-free.app
#   Linux/Mac: export BASE_URL=https://abc123.ngrok-free.app
#
APP_BASE_URL = os.environ.get("BASE_URL", "")


def get_base_url():
    """
    Returns the correct base URL for building password reset links.
    Priority:
      1. BASE_URL env var (if manually set)
      2. Auto-detected from the incoming request host
         Works for: localhost, LAN IP, ngrok, any deployed domain.
    """
    if APP_BASE_URL:
        return APP_BASE_URL.rstrip("/")
    return request.host_url.rstrip("/")


# ------------------ LOAD MODEL ------------------

try:
    print("Loading EfficientNet-B4 model...")
    model_tuple = load_efficientnet(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Model load failed. Train model first.")
    print("Error:", e)
    model_tuple = None


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

    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image TEXT,
            predicted_label TEXT,
            was_correct TEXT,
            actual_label TEXT,
            notes TEXT,
            rating INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expiry DATETIME NOT NULL,
            used INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()


init_db()


# ------------------ HELPER FUNCTIONS ------------------

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


# ------------------ RESET TOKEN HELPERS ------------------

def create_reset_token(email):
    """Generate a secure token, save to DB, return the token."""
    token = secrets.token_urlsafe(32)
    expiry = datetime.now() + timedelta(minutes=30)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Delete any existing unused tokens for this email
    c.execute("DELETE FROM password_reset_tokens WHERE email=?", (email,))

    c.execute(
        "INSERT INTO password_reset_tokens (email, token, expiry) VALUES (?, ?, ?)",
        (email, token, expiry.isoformat())
    )
    conn.commit()
    conn.close()
    return token


def verify_reset_token(token):
    """
    Check if token exists, is unused, and hasn't expired.
    Returns the email if valid, None otherwise.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT email, expiry, used FROM password_reset_tokens WHERE token=?",
        (token,)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return None  # Token not found

    email, expiry, used = row

    if used:
        return None  # Token already used

    if datetime.now() > datetime.fromisoformat(expiry):
        return None  # Token expired

    return email


def mark_token_used(token):
    """Mark token as used after password reset."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE password_reset_tokens SET used=1 WHERE token=?", (token,))
    conn.commit()
    conn.close()


# ------------------ EMAIL FUNCTION ------------------

def send_reset_email(to_email, reset_link):
    """Send a password reset link to the user's registered email."""
    msg = MIMEMultipart("alternative")
    msg['Subject'] = "DL Cattle Classifier - Password Reset"
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    text_body = f"""
Hi,

You requested a password reset for your DL Cattle Classifier account.

Click the link below to reset your password (valid for 30 minutes):

{reset_link}

If you did not request this, please ignore this email.
Your password will remain unchanged.

- DL Cattle Classifier Team
"""

    html_body = f"""
<html>
  <body style="font-family: Arial, sans-serif; color: #333;">
    <h2>Password Reset Request</h2>
    <p>You requested a password reset for your <strong>DL Cattle Classifier</strong> account.</p>
    <p>Click the button below to reset your password. This link is valid for <strong>30 minutes</strong>.</p>
    <p style="margin: 24px 0;">
      <a href="{reset_link}"
         style="background-color:#4CAF50; color:white; padding:12px 24px;
                text-decoration:none; border-radius:4px; font-size:16px;">
        Reset Password
      </a>
    </p>
    <p>Or copy and paste this link into your browser:</p>
    <p style="color:#555;">{reset_link}</p>
    <hr>
    <p style="font-size:12px; color:#999;">
      If you did not request this, you can safely ignore this email.
    </p>
  </body>
</html>
"""

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.ehlo()
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())


# ------------------ AUTH ROUTES ------------------

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


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------ FORGOT PASSWORD WITH RESET LINK ------------------

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        user = get_user_by_email(email)

        # ✅ Always show success message even if email not found
        # (prevents email enumeration attacks)
        if user:
            token = create_reset_token(email)

            # ✅ FIXED: auto-detects the correct host from the request.
            # Works for localhost, LAN IP, ngrok, and deployed domains
            # without any manual configuration.
            reset_link = f"{get_base_url()}/reset-password/{token}"

            # Prints to terminal so you can confirm the correct URL is used
            print(f"[DEBUG] Reset link sent: {reset_link}")

            try:
                send_reset_email(email, reset_link)
            except Exception as e:
                print("Email send error:", e)
                return render_template("forgot_password.html",
                                       error="Failed to send email. Please try again.")

        # Show success even if email doesn't exist (security best practice)
        return render_template("forgot_password.html",
                               success="If that email is registered, a reset link has been sent.")

    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    # ✅ Verify token is valid and not expired
    email = verify_reset_token(token)

    if not email:
        return render_template("reset_password.html",
                               error="This reset link is invalid or has expired. Please request a new one.")

    if request.method == "POST":
        new_password = request.form.get("password")

        if not new_password or len(new_password) < 6:
            return render_template("reset_password.html",
                                   token=token,
                                   error="Password must be at least 6 characters.")

        # ✅ Update password in DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE users SET password_hash=? WHERE email=?",
            (generate_password_hash(new_password), email)
        )
        conn.commit()
        conn.close()

        # ✅ Mark token as used so it can't be reused
        mark_token_used(token)

        return redirect(url_for("login"))

    return render_template("reset_password.html", token=token)


# ------------------ PROTECTED ROUTES ------------------

@app.route("/index")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/about")
def about():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if "user_id" not in session:
        return redirect(url_for("login"))

    submitted = False

    if request.method == "POST":
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            INSERT INTO feedback 
            (user_id, image, predicted_label, was_correct,
             actual_label, notes, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            request.form.get("image"),
            request.form.get("predicted"),
            request.form.get("correct"),
            request.form.get("actual_label"),
            request.form.get("notes"),
            request.form.get("rating")
        ))

        conn.commit()
        conn.close()
        submitted = True

    return render_template("feedback.html", submitted=submitted)


# ------------------ SERVE UPLOADS ------------------

@app.route("/uploads_dl/<path:filename>")
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD, filename)


# ------------------ PREDICTION ------------------

@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return {"error": "Unauthorized"}, 401

    if model_tuple is None:
        return {"error": "Model not loaded. Train first."}, 500

    file = request.files.get("image")
    url_input = request.form.get("image_url", "").strip()

    image_input = None
    image_url = None

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

    try:
        # Step 1: Detect cattle in the image first
        crop_path = detect_cattle(image_input)
        if crop_path is None:
            return {"error": "No cattle detected in the image. Please upload a valid cattle photo."}, 400

        # Step 2: Classify the detected crop
        labels, probs = predict_efficientnet(crop_path, model_tuple, topk=3)

        results = []
        max_prob = probs[0]

        # Step 3: Reject if confidence is too low
        if max_prob < 0.30:
            return {"error": "Cattle Undetected. Please upload the image of Cattle."}, 400

        if max_prob > 0.80:
            k = 1
        elif max_prob > 0.50:
            k = 2
        else:
            k = 3

        for i in range(k):
            label_clean = labels[i].strip().replace("_", " ").title()

            breed_details = next(
                (v for k2, v in BREED_INFO.items()
                 if k2.lower() == label_clean.lower()),
                {}
            )

            results.append({
                "breed": label_clean,
                "probability": round(probs[i] * 100, 2),
                "characteristics": breed_details.get("characteristics", "Not available"),
                "origin": breed_details.get("origin", "Not available"),
                "milk_yield": breed_details.get("milk_yield", "Not available"),
                "color": breed_details.get("color", "Not available")
            })

        return {
            "predictions": results,
            "image_path": image_url
        }

    except Exception as e:
        print("Prediction Error:", e)
        return {"error": "Prediction failed", "detail": str(e)}, 500


# ------------------ RUN APP ------------------

if __name__ == "__main__":
    # host="0.0.0.0" makes the app accessible on your LAN (e.g. 192.168.x.x)
    # so other devices on the same WiFi can reach it directly
    app.run(host="0.0.0.0", port=5002, debug=True)