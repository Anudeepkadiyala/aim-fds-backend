from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from lime.lime_text import LimeTextExplainer

router = APIRouter()

# Load pipeline that includes both vectorizer + model
model = joblib.load("models/hate_speech_model.pkl")

class HateSpeechInput(BaseModel):
    text: str

@router.post("/predict_hate")
def predict_hate(data: HateSpeechInput):
    text = data.text

    # Predict
    prediction = model.predict([text])[0]

    # Explain with LIME using the pipeline
    explainer = LimeTextExplainer(class_names=["Non-Hate", "Hate"])
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda x: model.predict_proba(x)
    )

    top_words = [word for word, weight in exp.as_list()][:5]

    return {
        "prediction": int(prediction),
        "explanation": top_words
    }
