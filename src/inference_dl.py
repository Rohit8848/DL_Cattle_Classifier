# src/inference_dl.py
import torch
from torchvision import transforms, models
from PIL import Image
import requests, io

IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485,0.456,0.406]
NORMALIZE_STD = [0.229,0.224,0.225]

def load_resnet(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model, ckpt['classes'], device

def preprocess_image(img):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    return tf(img).unsqueeze(0)

def load_image_from_path(path):
    return Image.open(path).convert('RGB')

def load_image_from_url(url):
    r = requests.get(url, timeout=8)
    img = Image.open(io.BytesIO(r.content)).convert('RGB')
    return img

def predict_resnet(image_input, model_tuple):
    model, classes, device = model_tuple
    if isinstance(image_input, str) and image_input.startswith("http"):
        img = load_image_from_url(image_input)
    else:
        img = load_image_from_path(image_input)
    x = preprocess_image(img).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)
        p, idx = torch.max(probs, dim=1)
    return classes[idx.item()], float(p.item())
