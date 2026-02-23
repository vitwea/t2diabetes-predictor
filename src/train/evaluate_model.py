import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
import numpy as np
import os

def evaluate_model():

    # -----------------------------
    # Load model + data
    # -----------------------------
    bundle = joblib.load("models/final_diabetes_model.pkl")
    model = bundle["model"]
    threshold = bundle["threshold"]

    X_train = pd.read_parquet("data/dataset/X_train.parquet")
    y_train = pd.read_parquet("data/dataset/y_train.parquet").squeeze()

    # -----------------------------
    # Predictions
    # -----------------------------
    probs = model.predict_proba(X_train)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------
    cm = confusion_matrix(y_train, y_pred)
    cm_norm = confusion_matrix(y_train, y_pred, normalize="true")

    roc_auc = roc_auc_score(y_train, probs)
    ap = average_precision_score(y_train, probs)

    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_train, probs)
    fpr, tpr, roc_thresholds = roc_curve(y_train, probs)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Confusion matrix
    axes[0, 0].imshow(cm, cmap="Blues")
    axes[0, 0].set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, cm[i, j], ha="center", va="center", fontsize=12)

    # 2. Normalized confusion matrix
    axes[0, 1].imshow(cm_norm, cmap="Blues")
    axes[0, 1].set_title("Normalized Confusion Matrix")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=12)

    # 3. Classification report
    report = classification_report(y_train, y_pred)
    axes[0, 2].axis("off")
    axes[0, 2].set_title("Classification Report")
    axes[0, 2].text(0, 0.5, report, fontsize=10, family="monospace")

    # 4. Precision-Recall curve
    axes[1, 0].plot(rec_curve, prec_curve, label=f"AP = {ap:.3f}")
    axes[1, 0].set_title("Precision-Recall Curve")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()

    # 5. ROC curve
    axes[1, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 1].set_title("ROC Curve")
    axes[1, 1].set_xlabel("False Positive Rate")
    axes[1, 1].set_ylabel("True Positive Rate")
    axes[1, 1].legend()

    # 6. Calibration curve
    prob_true, prob_pred = calibration_curve(y_train, probs, n_bins=10)
    axes[1, 2].plot(prob_pred, prob_true, marker="o")
    axes[1, 2].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 2].set_title("Calibration Curve")
    axes[1, 2].set_xlabel("Mean predicted probability")
    axes[1, 2].set_ylabel("Fraction of positives")

    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/model_evaluation.png", dpi=300)
    print("Imagen guardada en reports/model_evaluation.png")

if __name__ == "__main__":
    evaluate_model()