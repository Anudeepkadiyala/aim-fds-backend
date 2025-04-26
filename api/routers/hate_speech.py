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
    try:
        text = data.text
        print("📥 Received input for Hate Speech:", text)

        # Predict
        prediction = model.predict([text])[0]
        print("✅ Model prediction:", prediction)

        # Explain with LIME
        explainer = LimeTextExplainer(class_names=["Non-Hate", "Hate"])
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=lambda x: model.predict_proba(x)
        )

        top_words = [word for word, weight in exp.as_list()][:5]
        print("💡 Explanation top words:", top_words)

        return {
            "prediction": int(prediction),
            "explanation": top_words
        }

    except Exception as e:
        print("❌ Error in /predict_hate:", e)
        return {
            "error": "Something went wrong during hate speech prediction.",
            "details": str(e)
        }
