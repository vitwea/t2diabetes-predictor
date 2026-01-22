"""
TRAIN BASELINE MODELS - Complete training pipeline
===================================================

Train 3 algorithms (LR, RF, XGBoost) on each of 3 datasets (M1, M2, M3)
Evaluate performance and save trained models
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
    accuracy_score
)
from imblearn.metrics import (sensitivity_score, specificity_score)
import xgboost as xgb
from src.utils.logger import get_logger

logger = get_logger("train_baseline_models")


def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical variables using LabelEncoder
    Returns: df_encoded, encoders dict
    """
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    return df_encoded, encoders


def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_name):
    """
    Train 3 models and evaluate each
    Returns: dict with results and trained models
    """
    
    results = {}
    models = {}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING 3 MODELS FOR {model_name}")
    logger.info(f"{'='*80}")
    logger.info(f"\nTrain set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    logger.info(f"Test set:  {X_test.shape[0]:,} samples")
    logger.info(f"Target prevalence: {y_train.sum() / len(y_train) * 100:.1f}%")
    
    # ===== MODEL 1: LOGISTIC REGRESSION =====
    logger.info(f"\n{'─'*80}")
    logger.info("MODEL 1: LOGISTIC REGRESSION")
    logger.info(f"{'─'*80}")
    
    # Standardize features for LR
    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train)
    X_test_scaled = scaler_lr.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
    sens_lr = tp / (tp + fn)  # Sensitivity
    spec_lr = tn / (tn + fp)  # Specificity
    
    logger.info(f"  AUC:          {auc_lr:.4f}")
    logger.info(f"  Accuracy:     {acc_lr:.4f}")
    logger.info(f"  Sensitivity:  {sens_lr:.4f}")
    logger.info(f"  Specificity:  {spec_lr:.4f}")
    
    results['logistic_regression'] = {
        'auc': auc_lr,
        'accuracy': acc_lr,
        'sensitivity': sens_lr,
        'specificity': spec_lr,
        'y_proba': y_proba_lr,
        'y_pred': y_pred_lr
    }
    models['logistic_regression'] = lr
    
    # ===== MODEL 2: RANDOM FOREST =====
    logger.info(f"\n{'─'*80}")
    logger.info("MODEL 2: RANDOM FOREST")
    logger.info(f"{'─'*80}")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    sens_rf = tp / (tp + fn)
    spec_rf = tn / (tn + fp)
    
    logger.info(f"  AUC:          {auc_rf:.4f}")
    logger.info(f"  Accuracy:     {acc_rf:.4f}")
    logger.info(f"  Sensitivity:  {sens_rf:.4f}")
    logger.info(f"  Specificity:  {spec_rf:.4f}")
    
    results['random_forest'] = {
        'auc': auc_rf,
        'accuracy': acc_rf,
        'sensitivity': sens_rf,
        'specificity': spec_rf,
        'y_proba': y_proba_rf,
        'y_pred': y_pred_rf,
        'feature_importance': rf.feature_importances_
    }
    models['random_forest'] = rf
    
    # ===== MODEL 3: XGBOOST =====
    logger.info(f"\n{'─'*80}")
    logger.info("MODEL 3: XGBOOST")
    logger.info(f"{'─'*80}")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=len(y_train) / (y_train.sum() + 1e-8)  # Handle imbalance
    )
    xgb_model.fit(X_train, y_train)
    
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    auc_xgb = roc_auc_score(y_test, y_proba_xgb)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
    sens_xgb = tp / (tp + fn)
    spec_xgb = tn / (tn + fp)
    
    logger.info(f"  AUC:          {auc_xgb:.4f}")
    logger.info(f"  Accuracy:     {acc_xgb:.4f}")
    logger.info(f"  Sensitivity:  {sens_xgb:.4f}")
    logger.info(f"  Specificity:  {spec_xgb:.4f}")
    
    results['xgboost'] = {
        'auc': auc_xgb,
        'accuracy': acc_xgb,
        'sensitivity': sens_xgb,
        'specificity': spec_xgb,
        'y_proba': y_proba_xgb,
        'y_pred': y_pred_xgb,
        'feature_importance': xgb_model.feature_importances_
    }
    models['xgboost'] = xgb_model
    
    # ===== SUMMARY =====
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Model Comparison")
    logger.info(f"{'='*80}")
    
    summary_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'AUC': [auc_lr, auc_rf, auc_xgb],
        'Accuracy': [acc_lr, acc_rf, acc_xgb],
        'Sensitivity': [sens_lr, sens_rf, sens_xgb],
        'Specificity': [spec_lr, spec_rf, spec_xgb]
    })
    
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    return results, models, summary_df, scaler_lr


def main():
    """
    Main pipeline: Load datasets, train models, save results
    """
    
    logger.info("="*80)
    logger.info("BASELINE MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # ===== LOAD DATASETS =====
    datasets = {
        'Model 1 (Population)': './data/final/nhanes_model1_population.parquet',
        'Model 2 (Metabolic)': './data/final/nhanes_model2_metabolic.parquet',
        'Model 3 (Deep Metabolic)': './data/final/nhanes_model3_deepmetabolic.parquet'
    }
    
    all_results = {}
    all_models = {}
    all_summaries = {}
    
    for model_label, filepath in datasets.items():
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# {model_label}")
        logger.info(f"{'#'*80}")
        
        try:
            df = pd.read_parquet(filepath)
            logger.info(f"\nLoaded: {len(df):,} rows × {df.shape[1]} columns")
            
            # ===== PREPARE DATA =====
            # Remove ID and target
            X = df.drop(['diabetes_dx', 'id'], axis=1, errors='ignore')
            y = df['diabetes_dx']
            
            logger.info(f"\nFeatures: {X.shape[1]}")
            logger.info(f"Target: {y.value_counts().to_dict()}")
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                logger.info(f"Categorical columns: {categorical_cols}")
                X, encoders = encode_categorical_features(X, categorical_cols)
            
            # Handle missing values (forward fill or median)
            logger.info(f"Missing values before imputation: {X.isna().sum().sum()}")
            X = X.fillna(X.median())
            logger.info(f"Missing values after imputation: {X.isna().sum().sum()}")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            logger.info(f"\nTrain-test split:")
            logger.info(f"  Train: {X_train.shape[0]:,} ({y_train.sum():,} positive)")
            logger.info(f"  Test:  {X_test.shape[0]:,} ({y_test.sum():,} positive)")
            
            # ===== TRAIN MODELS =====
            results, models, summary_df, scaler = train_and_evaluate_models(
                X_train, X_test, y_train, y_test, model_label
            )
            
            all_results[model_label] = results
            all_models[model_label] = {
                'models': models,
                'scaler': scaler,
                'features': X.columns.tolist(),
                'X_test': X_test,
                'y_test': y_test
            }
            all_summaries[model_label] = summary_df
            
            # ===== SAVE MODELS =====
            model_dir = './models'
            import os
            os.makedirs(model_dir, exist_ok=True)
            
            for algo_name, model in models.items():
                model_path = f'{model_dir}/{model_label.replace(" ", "_")}_{algo_name}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"\n✓ Saved: {model_path}")
            
            # Save scaler
            scaler_path = f'{model_dir}/{model_label.replace(" ", "_")}_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"✓ Saved: {scaler_path}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {model_label}: {e}")
    
    # ===== FINAL SUMMARY ACROSS ALL MODELS =====
    logger.info(f"\n\n{'='*80}")
    logger.info("OVERALL SUMMARY: All Models Comparison")
    logger.info(f"{'='*80}")
    
    for model_label, summary_df in all_summaries.items():
        logger.info(f"\n{model_label}:")
        logger.info(f"\n{summary_df.to_string(index=False)}")
    
    logger.info(f"\n{'='*80}")
    logger.info("✓ BASELINE MODEL TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info("""

NEXT STEPS:
───────────
1. ✓ Models trained and saved in ./models/

2. ⏭️  Create evaluate_shap_importance.py
   - Load trained models
   - Calculate SHAP values for feature importance
   - Determine which features actually matter

3. ⏭️  Create generate_final_report.py
   - Create publication-ready tables
   - ROC curves for each model
   - Feature importance plots

""")


if __name__ == "__main__":
    main()