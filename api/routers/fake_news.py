from fastapi import APIRouter
from pydantic import BaseModel
import joblib

router = APIRouter()

# Load model
model = joblib.load("models/fake_news_model.pkl")

class NewsInput(BaseModel):
    text: str

@router.post("/predict_news")
def predict_news(data: NewsInput):
    prediction = model.predict([data.text])[0]
    return {"prediction": int(prediction)}
