from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

router = APIRouter()

# Load model
model = joblib.load("models/fraud_model.pkl")

# Correct feature names (30 columns)
feature_names = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28", "Amount"
]

class FraudInput(BaseModel):
    features: list

@router.post("/predict_fraud")
async def predict_fraud(data: FraudInput):
    input_array = np.array(data.features).reshape(1, -1)

    # ✨ Validate input
    if input_array.shape[1] != 30:
        return {"error": f"Expected 30 features, but got {input_array.shape[1]}"}

    # ✅ Predict
    prediction = model.predict(input_array)[0]

    # ✅ LIME Explainability
    explainer = LimeTabularExplainer(
        training_data=np.zeros((1, len(feature_names))),
        feature_names=feature_names,
        class_names=["Non-Fraud", "Fraud"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        data_row=input_array[0],
        predict_fn=lambda x: model.predict_proba(x)
    )

    important_features = [feature for feature, weight in exp.as_list()][:5]

    return {
        "prediction": int(prediction),
        "explanation": important_features
    }
