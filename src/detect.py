from ultralytics import YOLO
import cv2
import os
import uuid
import requests
import numpy as np

model = YOLO("yolov8n.pt")
model.to("cpu")

def detect_cattle(image_path):
    # --- Handle URL input: download to a temp local file ---
    is_url = image_path.startswith("http://") or image_path.startswith("https://")
    if is_url:
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        arr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Save to a temp local path so YOLO and cv2 can work with it
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"temp_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(temp_path, image)
        local_path = temp_path
    else:
        local_path = image_path
        image = cv2.imread(local_path)

    # --- Run YOLO detection ---
    results = model(local_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # Clean up temp file if URL was used
    if is_url and os.path.exists(temp_path):
        os.remove(temp_path)

    if len(boxes) == 0:
        return None

    # Crop the first detected cattle
    x1, y1, x2, y2 = map(int, boxes[0])
    crop = image[y1:y2, x1:x2]

    # Save crop next to original (or in same folder for URL case)
    base, ext = os.path.splitext(image_path if not is_url else local_path)
    # For URLs, save crop in script directory
    if is_url:
        crop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"crop_{uuid.uuid4().hex}.jpg")
    else:
        crop_path = base + "_crop" + ext

    cv2.imwrite(crop_path, crop)
    return crop_path