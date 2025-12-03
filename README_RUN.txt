# create venv and install
python -m venv venv
# Windows: venv\Scripts\activate
# mac/linux: source venv/bin/activate
pip install -r requirements.txt

# Prepare split (images must be in dataset/images/<class>/*.jpg)
python dataset/prepare_split.py

# Train ResNet-50
python src/train_dl.py --data dataset/split --out models/resnet50_checkpoint.pth --epochs 8 --batch 32 --lr 1e-4

# Run DL web app
python src/app_dl.py 
# app runs on http://127.0.0.1:5002



while re running the program
venv\Scripts\activate.bat
pip install -r requirements.txt
python -m src.app_dl



Open PowerShell in your project directory.

Run the SQLite3 CLI on your database file:
sqlite3 src/users.db

Inside the SQLite prompt, check all user entries:
SELECT * FROM users;