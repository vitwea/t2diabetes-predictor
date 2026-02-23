from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Import your preprocessing pipeline
from src.pipeline.run_preprocessing import preprocessing, load_artifacts


# -------------------------------------------------------------------
# Load model bundle (model + threshold + params)
# -------------------------------------------------------------------
def load_model_bundle(path="models/final_diabetes_model.pkl"):
    bundle = joblib.load(path)
    model = bundle["model"]
    threshold = bundle["threshold"]
    params = bundle.get("best_params", {})
    return model, threshold, params


# -------------------------------------------------------------------
# Preprocess input (TRANSFORM mode)
# -------------------------------------------------------------------
def preprocess_input(df_raw):
    artifacts = load_artifacts(Path("data/dataset/preprocessing_artifacts.joblib"))

    encoding_info = artifacts.get("encoding_info", {})
    encoding_config = encoding_info.get("encoding_config", {})

    df_proc = preprocessing(
        df_raw,
        encoding_config=encoding_config,
        select_top=True,
        mode="transform",
        artifacts=artifacts,
        target_col="diabetes_risk"
    )

    return df_proc


# -------------------------------------------------------------------
# Prediction function
# -------------------------------------------------------------------
def predict(df_raw):
    df_proc = preprocess_input(df_raw)

    model, threshold, params = load_model_bundle()

    probs = model.predict_proba(df_proc)[:, 1]
    preds = (probs >= threshold).astype(int)

    return preds, probs, threshold, params


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="API para predecir riesgo de diabetes tipo 2 usando modelo calibrado",
    version="1.0"
)


# -------------------------------------------------------------------
# Request schema
# -------------------------------------------------------------------
class PatientData(BaseModel):
    # Puedes añadir validaciones opcionales aquí
    data: dict


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.post("/predict")
def predict_endpoint(payload: PatientData):
    df_raw = pd.DataFrame([payload.data])

    preds, probs, threshold, params = predict(df_raw)

    return {
        "prediction": int(preds[0]),
        "probability": float(probs[0]),
        "threshold_used": float(threshold),
        "model_params": params
    }