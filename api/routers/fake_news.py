from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from lime.lime_text import LimeTextExplainer

router = APIRouter()

# ✅ Load the full pipeline (tfidf + logistic regression)
model = joblib.load("models/fake_news_model.pkl")

class NewsInput(BaseModel):
    text: str

@router.post("/predict_news")
def predict_news(data: NewsInput):
    text = data.text

    # ✅ Predict directly with pipeline
    prediction = model.predict([text])[0]

    # ✅ Explainability using LIME on pipeline
    explainer = LimeTextExplainer(class_names=["Real", "Fake"])

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda x: model.predict_proba(x)
    )

    # ✅ Top 5 influential words
    top_words = [word for word, weight in exp.as_list()][:5]

    return {
        "prediction": int(prediction),
        "explanation": top_words
    }
