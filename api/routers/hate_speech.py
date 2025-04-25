from fastapi import APIRouter
from pydantic import BaseModel
import joblib

router = APIRouter()

# Load model
model = joblib.load("models/hate_speech_model.pkl")

class HateSpeechInput(BaseModel):
    text: str

@router.post("/predict_hate")
def predict_hate(data: HateSpeechInput):
    prediction = model.predict([data.text])[0]
    return {"prediction": int(prediction)}
