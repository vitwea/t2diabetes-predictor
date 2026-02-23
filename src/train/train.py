import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from scipy.stats import uniform, randint

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from src.utils.logger import get_logger
logger = get_logger("train")

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------

def load_data():
    X_train = pd.read_parquet("data/dataset/X_train.parquet")
    y_train = pd.read_parquet("data/dataset/y_train.parquet").squeeze()
    return X_train, y_train

# -------------------------------------------------------------------
# 2. Threshold selection
# -------------------------------------------------------------------

def find_best_threshold(y_true, y_prob, min_recall=0.85):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_t, best_score = None, -1
    best_p, best_r = None, None

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 2 * (p * r) / (p + r + 1e-9)
        if r >= min_recall and f1 > best_score:
            best_score = f1
            best_t = t
            best_p = p
            best_r = r

    # fallback: best F1 if no threshold meets recall target
    if best_t is None:
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        idx = np.nanargmax(f1_scores)
        best_t = thresholds[idx]
        best_p = precision[idx]
        best_r = recall[idx]

    return best_t, best_p, best_r

# -------------------------------------------------------------------
# 3. Train + tuning + calibration
# -------------------------------------------------------------------

def train_model(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = HistGradientBoostingClassifier(random_state=42)

    # -----------------------------
    # RandomizedSearch
    # -----------------------------
    param_dist = {
        "learning_rate": uniform(0.01, 0.3),
        "max_iter": randint(100, 1000),
        "max_leaf_nodes": randint(15, 255),
        "min_samples_leaf": randint(1, 50),
        "l2_regularization": uniform(0.0, 1.0)
    }

    rs = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rs.fit(X_train, y_train)

    best_rs = rs.best_params_

    # -----------------------------
    # GridSearch (refinement)
    # -----------------------------
    param_grid = {
        "learning_rate": sorted({max(0.001, best_rs["learning_rate"] * f) for f in [0.5, 1.0, 1.5]}),
        "max_iter": sorted({max(50, best_rs["max_iter"] + d) for d in [-100, 0, 100]}),
        "max_leaf_nodes": sorted({max(2, best_rs["max_leaf_nodes"] + d) for d in [-32, 0, 32]})
    }

    gs = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_

    # -----------------------------
    # Calibration
    # -----------------------------
    cal = CalibratedClassifierCV(best_model, method="isotonic", cv=5)
    cal.fit(X_train, y_train)

    # -----------------------------
    # Threshold selection
    # -----------------------------
    probs = cal.predict_proba(X_train)[:, 1]
    threshold, prec, rec = find_best_threshold(y_train, probs, min_recall=0.85)

    # -----------------------------
    # MÉTRICAS AÑADIDAS
    # -----------------------------
    roc_auc = roc_auc_score(y_train, probs)
    ap = average_precision_score(y_train, probs)

    y_pred = (probs >= threshold).astype(int)

    f1 = f1_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    cm = confusion_matrix(y_train, y_pred)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Average Precision (AP): {ap:.4f}")
    logger.info(f"F1 (threshold={threshold:.3f}): {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Confusion matrix:\n{cm}")

    return cal, threshold, gs.best_params_

# -------------------------------------------------------------------
# 4. Save final model
# -------------------------------------------------------------------

def save_model(model, threshold, params):
    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": model,
        "threshold": threshold,
        "best_params": params
    }
    joblib.dump(bundle, "models/final_diabetes_model.pkl")
    logger.info("Model saved to models/final_diabetes_model.pkl")

# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Loading data...")
    X_train, y_train = load_data()

    logger.info("Training model with tuning + calibration...")
    model, threshold, params = train_model(X_train, y_train)

    logger.info(f"Best threshold: {threshold:.3f}")
    logger.info("Best hyperparameters: %s", params)

    save_model(model, threshold, params)

    logger.info("Training complete.")