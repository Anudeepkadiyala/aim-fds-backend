import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
import os
MODEL_PATH = os.path.join("models", "deepfake", "deepfake_model.h5")
model = load_model(MODEL_PATH)


def predict_image(img_array):
    prediction = model.predict(img_array)
    result = "Fake" if prediction[0][0] > 0.5 else "Real"
    return {
        "prediction": result,
        "confidence": float(prediction[0][0])
    }
