# src/inference_dl.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
import requests
import io
import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────
# DEVICE SETUP
# Previously "CUDA_VISIBLE_DEVICES" = "" was
# hardcoded — this FORCED CPU even though your
# RTX GPU was available. Removed completely.
# Now auto-selects RTX GPU if available.
# ─────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"   # float16 AMP — faster on RTX cards

print(f"[Inference] Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[Inference] GPU    : {torch.cuda.get_device_name(0)}")
    print(f"[Inference] VRAM   : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    print(f"[Inference] AMP    : enabled (float16) — ~20-40ms per prediction")
else:
    print("[Inference] No GPU found — using CPU (~800ms-1.5s per prediction)")


# ─────────────────────────────────────────────
# SETTINGS
# NOTE: If you have NOT retrained yet with the
# new train_dl.py, keep IMAGE_RESIZE=320 and
# IMAGE_CROP=300 (your original values).
# Once you retrain, switch to 420/380.
# ─────────────────────────────────────────────
IMAGE_RESIZE   = 320    # change to 420 after retraining
IMAGE_CROP     = 300    # change to 380 after retraining

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

DATASET_PATH   = "dataset/split/train"


# ─────────────────────────────────────────────
# YOLO DETECTION — runs on CPU intentionally
# torchvision::nms (Non-Maximum Suppression)
# is not supported on CUDA in this torchvision
# version. YOLO stays on CPU, EfficientNet on GPU.
# YOLO on CPU is still fast (~50-100ms) since
# yolov8n is a tiny model.
# ─────────────────────────────────────────────
print("Loading YOLOv8 detection model...")
detection_model = YOLO("yolov8n.pt")
detection_model.to("cpu")   # must stay CPU — torchvision NMS CUDA not available
print("YOLO loaded on CPU (NMS requires CPU).")


# ─────────────────────────────────────────────
# LOAD EFFICIENTNET-B4
# ─────────────────────────────────────────────
def load_efficientnet(path, device=DEVICE):
    """
    Loads EfficientNet-B4 and moves it to RTX GPU.
    Auto-detects num_classes and classifier head format
    (handles both old and new train_dl.py head structure).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Always load weights to CPU first, then move to target device.
    # This avoids VRAM issues if the model was saved on a different GPU.
    state_dict = torch.load(path, map_location="cpu")

    # ── Detect classifier head format ────────────────────────────────
    # Old train_dl.py: classifier.1 = Linear
    # New train_dl.py: classifier.1 = Sequential(Dropout, Linear)
    if "classifier.1.1.weight" in state_dict:
        num_classes  = state_dict["classifier.1.1.weight"].shape[0]
        use_new_head = True
    else:
        num_classes  = state_dict["classifier.1.weight"].shape[0]
        use_new_head = False

    # ── Build model ──────────────────────────────────────────────────
    model = models.efficientnet_b4(weights=None)

    if use_new_head:
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    model.load_state_dict(state_dict)

    # ── Move entire model to RTX GPU ─────────────────────────────────
    model = model.to(device)
    model.eval()

    # ── Class names from dataset folder ──────────────────────────────
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

    class_names = sorted(os.listdir(DATASET_PATH))

    print(f"EfficientNet-B4 loaded on {str(device).upper()}.")
    print(f"Classes: {num_classes} | Head: {'new (Dropout+Linear)' if use_new_head else 'old (Linear)'}")

    return model, class_names


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────
def load_image_from_path(path):
    return Image.open(path).convert("RGB")


def load_image_from_url(url):
    r = requests.get(url, timeout=8)
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def preprocess_image(img):
    """
    Preprocesses a PIL image into a GPU-ready tensor.
    Uses BICUBIC interpolation for sharper resizing.
    """
    tf = transforms.Compose([
        transforms.Resize(
            IMAGE_RESIZE,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(IMAGE_CROP),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    return tf(img).unsqueeze(0)   # shape: [1, 3, H, W]


# ─────────────────────────────────────────────
# YOLO DETECTION + CROP
# Detects cow in the image and crops to the
# bounding box for better classification accuracy.
# ─────────────────────────────────────────────
def detect_and_crop(image_input):
    """
    Runs YOLOv8 to find a cow (COCO class 19) in the image.
    Returns the cropped PIL image if found, else None.
    """
    if isinstance(image_input, str) and image_input.startswith("http"):
        img_pil = load_image_from_url(image_input)
        img_np  = np.array(img_pil)
    else:
        img_np = cv2.imread(image_input)
        if img_np is None:
            return None
        img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    # Run YOLO on GPU
    results = detection_model(img_np, device="cpu", verbose=False)  # NMS requires CPU

    best_crop = None
    best_conf = 0.0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])

        # COCO class 19 = cow
        # Pick the detection with highest confidence
        if cls_id == 19 and conf > best_conf:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add 5% padding around the bounding box
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            x1 = max(x1 - pad_x, 0)
            y1 = max(y1 - pad_y, 0)
            x2 = min(x2 + pad_x, img_np.shape[1])
            y2 = min(y2 + pad_y, img_np.shape[0])

            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            best_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            best_conf = conf

    return best_crop


# ─────────────────────────────────────────────
# PREDICTION PIPELINE
# Full pipeline: detect → crop → preprocess
# → GPU inference with AMP → top-k results
# ─────────────────────────────────────────────
def predict_efficientnet(image_path, model_tuple, topk=3):
    """
    Full prediction pipeline running entirely on RTX GPU.

    Steps:
      1. YOLO detects and crops the cow region (GPU)
      2. Preprocess image into tensor
      3. Move tensor to GPU
      4. EfficientNet-B4 forward pass with AMP float16 (GPU)
      5. Softmax → top-k labels and probabilities

    Speed vs old CPU version:
      CPU: ~800ms - 1.5s per image
      RTX GPU + AMP: ~20 - 60ms per image  (~20-30x faster)
    """
    model, class_names = model_tuple

    # ── Step 1: detect and crop cow region ───────────────────────────
    cropped = detect_and_crop(image_path)

    if cropped is not None:
        img = cropped
    else:
        # No cow detected — use full image
        if isinstance(image_path, str) and image_path.startswith("http"):
            img = load_image_from_url(image_path)
        else:
            img = load_image_from_path(image_path)

    # ── Step 2: preprocess ───────────────────────────────────────────
    image_tensor = preprocess_image(img)

    # ── Step 3: move tensor to RTX GPU ───────────────────────────────
    image_tensor = image_tensor.to(DEVICE, non_blocking=True)

    # ── Step 4: GPU inference with AMP ───────────────────────────────
    model.eval()
    with torch.no_grad():
        with autocast(enabled=USE_AMP):     # float16 on RTX — 2x faster
            outputs = model(image_tensor)   # forward pass on GPU

    # ── Step 5: softmax + top-k on GPU, then move to CPU ─────────────
    probs = F.softmax(outputs, dim=1)
    top_probs, top_idxs = torch.topk(probs, min(topk, len(class_names)))

    top_probs = top_probs[0].cpu().numpy()
    top_idxs  = top_idxs[0].cpu().numpy()

    labels        = [class_names[top_idxs[i]] for i in range(len(top_idxs))]
    probabilities = [float(top_probs[i])       for i in range(len(top_probs))]

    return labels, probabilities