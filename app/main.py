from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur de chargement modèle : {e}")

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
