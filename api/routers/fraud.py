from fastapi import APIRouter, File, UploadFile
import pandas as pd
import joblib
import tempfile

router = APIRouter()

# Load model
model = joblib.load("models/fraud_model.pkl")

@router.post("/predict_fraud")
async def predict_fraud(file: UploadFile = File(...)):
    contents = await file.read()
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    df = pd.read_csv(tmp_path)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
