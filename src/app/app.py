import argparse
import json
import joblib
import pandas as pd
import numpy as np
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
# Apply preprocessing (TRANSFORM mode)
# -------------------------------------------------------------------
def preprocess_input(df_raw):
    artifacts = load_artifacts(Path("data/dataset/preprocessing_artifacts.joblib"))

    # Load encoding config from artifacts
    encoding_info = artifacts.get("encoding_info", {})
    encoding_config = encoding_info.get("encoding_config", {})

    # Apply preprocessing in TRANSFORM mode
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
# Predict function
# -------------------------------------------------------------------
def predict(df_raw):
    # Preprocess
    df_proc = preprocess_input(df_raw)

    # Load model + threshold
    model, threshold, params = load_model_bundle()

    # Predict probabilities
    probs = model.predict_proba(df_proc)[:, 1]

    # Apply threshold
    preds = (probs >= threshold).astype(int)

    return preds, probs, threshold


# -------------------------------------------------------------------
# CLI interface
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict diabetes risk")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to CSV or JSON file with raw input data")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save predictions to CSV")

    args = parser.parse_args()

    # Load input
    if args.input.endswith(".csv"):
        df_raw = pd.read_csv(args.input)
    elif args.input.endswith(".json"):
        with open(args.input, "r") as f:
            data = json.load(f)
        df_raw = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
    else:
        raise ValueError("Input must be CSV or JSON")

    preds, probs, threshold = predict(df_raw)

    df_out = df_raw.copy()
    df_out["probability"] = probs
    df_out["prediction"] = preds
    df_out["threshold_used"] = threshold

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(df_out)


if __name__ == "__main__":
    main()