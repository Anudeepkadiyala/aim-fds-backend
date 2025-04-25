from fastapi import APIRouter, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import os
import shutil
import cv2
import uuid
from utils.video_utils import extract_frames  # 🔁 We'll create this file

router = APIRouter()

# ✅ Load your trained model (adjust path if needed)
model_path = os.path.join("models", "deepfake", "deepfake_model.h5")
model = load_model(model_path)

# ------------------------------
# 📸 Image Deepfake Detection
# ------------------------------
@router.post("/deepfake/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize((128, 128))  # ✅ Match training input size

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        result = "Fake" if prediction > 0.5 else "Real"

        return {
            "filename": file.filename,
            "prediction": result,
            "confidence": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# 🎥 Video Deepfake Detection
# ------------------------------
@router.post("/deepfake/video-detect")
async def detect_deepfake_in_video(file: UploadFile = File(...)):
    try:
        temp_video_path = f"uploads/videos/{file.filename}"
        os.makedirs("uploads/videos", exist_ok=True)

        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Clean and prepare frame directory
        frame_dir = "uploads/frames"
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)

        # Extract frames
        frame_paths = extract_frames(temp_video_path, frame_dir)

        results = []
        for frame_path in frame_paths:
            img = image.load_img(frame_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0][0]
            results.append(pred)

        fake_count = sum(p > 0.5 for p in results)
        real_count = len(results) - fake_count

        return {
            "filename": file.filename,
            "total_frames": len(results),
            "fake_frames": fake_count,
            "real_frames": real_count,
            "verdict": "Fake" if fake_count > real_count else "Real"
        }

    except Exception as e:
        return {"error": str(e)}
