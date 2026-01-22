# T2 Diabetes Predictor: Machine Learning Pipeline

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)](#)

A comprehensive machine learning pipeline for **Type 2 Diabetes prediction** using NHANES clinical data, featuring enhanced feature engineering with HDL estimation and production-ready data preparation.

## 🎯 Overview

This project implements a complete ML pipeline for diabetes prediction:

```
Raw Data 
    ↓
Feature Engineering
    ↓
Engineered Data 
    ↓
Data Cleaning 
    ↓
Clean Data 
    ↓
Data Preparation 
    ↓
ML-Ready Data
    ↓
Model Training & Evaluation
```

## ✨ Key Features

### 🧬 Enhanced Feature Engineering
- **23 Clinical Features** including:
  - Blood Pressure indices (4): MAP, Pulse Pressure, Systolic, Diastolic
  - Insulin Resistance (2): HOMA-IR, QUICKI
  - Anthropometric (2): Waist-Height ratio, BMI-Waist ratio
  - Advanced lipids (3): TyG, TyG-Waist, Non-HDL
  - Diet composition (4): Carb%, Fat%, Protein%, ratios
  - Metabolic Syndrome Score (complete with HDL)
  - Cardiovascular stress indicator

### 🔧 Robust Data Pipeline
- **Stratified train/test split** (80/20) preserving class distribution
- **SMOTE resampling** for class balance (50/50 in training)
- **StandardScaler normalization** (μ=0, σ=1)
- **Inf/-inf handling** with NaN conversion before imputation
- **Parquet persistence** for reproducibility
- Comprehensive **metadata tracking** (feature names, shapes, distributions)

### 📊 Production Quality
- Full logging and error handling
- Type conversion and validation
- Outlier detection (IQR method)
- Sparse column dropping (>50% NaN)
- JSON metadata export for auditability

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/vitwea/t2diabetes-predictor.git
cd t2diabetes-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 🚀 Quick Start

### 1. Feature Engineering

```bash
python -m src.data.modeling.feature_engineer
```

**Output**: `nhanes_diabetes_engineered.parquet` (57,395 × 46)

Creates 23 engineered features.

### 2. Data Preparation

```bash
python -m src.modeling.main
```

**Outputs**:
- `train_prepared.parquet` (79,284 × 45) - Scaled & balanced
- `test_prepared.parquet` (10,876 × 45) - Scaled
- `prep_metadata.json` - Feature names & metadata

### 3. Train Models

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Load prepared data
train_df = pd.read_parquet("./data/final/train_prepared.parquet")
test_df = pd.read_parquet("./data/final/test_prepared.parquet")

# Prepare X, y
X_train = train_df.drop(columns=['diabetes_dx']).values
y_train = train_df['diabetes_dx'].values
X_test = test_df.drop(columns=['diabetes_dx']).values
y_test = test_df['diabetes_dx'].values

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.4f}")
print(classification_report(y_test, model.predict(X_test)))
```

## 📁 Project Structure

```
t2diabetes-predictor/
│
├── src/
│   ├── data/
│   │   └── modeling/
│   │       ├── feature_engineer_nhanes.py    
│   │       └── data_cleaner.py                        
│   │
│   ├── modeling/
│   │   ├── main.py                    # Entry point
│   │   └── pipeline.py                # Data preparation pipeline
│   │
│   └── utils/
│       └── logger.py                  # Logging utility
│
├── notebooks/
│   └── 01_eda.ipynb      
|           
├── docs/
│   └── README_DATA.md                 # Data documentation
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── PARQUET_SETUP_GUIDE.md            # Detailed setup guide
├── ENHANCED_vs_ORIGINAL.md           # Feature comparison
├── HDL_ESTIMATION_GUIDE.md           # HDL methodology
└── GIT_COMMITS_PLAN.md               # Git strategy
```

## 📊 Data Overview

### Input
- **Dataset**: NHANES (National Health and Nutrition Examination Survey)
- **Samples**: 57,395
- **Raw Features**: 24 (age, glucose, BP, lipids, anthropometrics, diet)

### Process
1. **Feature Engineering**: +23 clinical features
2. **Cleaning**: Remove 3-5% outliers, validate ranges
3. **Preparation**: Stratified split → Imputation → SMOTE → Scaling

### Output (ML-Ready)
- **Training**: 79,284 × 44 (scaled, balanced 50/50)
- **Testing**: 10,876 × 44 (scaled, original distribution ~8.9% positive)
- **Target**: Binary (0=No Diabetes, 1=Type 2 Diabetes)

### Features (44 total)

| Category | Count | Examples |
|----------|-------|----------|
| Blood Pressure | 4 | MAP, Pulse Pressure |
| Insulin Resistance | 2 | HOMA-IR, QUICKI |
| Anthropometric | 2 | Waist-Height, BMI-Waist |
| Glucose/Lipid | 2 | Glucose-HbA1c, TG-Chol |
| Advanced Lipids | 3 | TyG, TyG-Waist, Non-HDL |
| Diet Composition | 4 | Carb%, Fat%, Protein% |
| CV Stress | 1 | Sys/Dia ratio |
| Metabolic Syndrome | 1 | MetS Score (0-5) |

## 🔬 Methodology

### Handling Class Imbalance
Original dataset: ~8.9% positive (diabetes), ~91.1% negative

**Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- Creates synthetic samples of minority class
- Result: 50/50 balanced training set
- Prevents model bias toward majority class
- Test set maintains original distribution for realistic evaluation

### Data Scaling
`StandardScaler` normalization:
- Mean: 0, Std Dev: 1
- Fit on training data
- Applied to test data (prevents leakage)
- Essential for algorithms sensitive to feature scale (LR, SVM, NN, tree-based)

## 📈 Expected Performance

### Baseline (Original 24 features)
- Estimated AUC: ~0.75-0.78

### With Enhanced Features (44 features + HDL estimation)
- Expected AUC: **~0.82-0.85** (+3-7% improvement)
- Better discrimination between diabetic/non-diabetic patients
- Improved feature importance distribution

## 🛠️ Usage Examples

### Example 1: Complete Pipeline Execution

```bash
# Feature engineering
python -m src.data.modeling.feature_engineer_nhanes_enhanced

# Data preparation
python -m src.modeling.main

# Now train models with prepared data
```

### Example 2: Inspect Prepared Data

```python
import pandas as pd
import json

# Load training data
train_df = pd.read_parquet("./data/final/train_prepared.parquet")
print(train_df.shape)  # (79284, 45)
print(train_df.describe())

# Load metadata
with open("./data/final/prep_metadata.json") as f:
    metadata = json.load(f)
    print(metadata['feature_names'])
    print(metadata['class_distribution_train'])
```

### Example 3: Train Multiple Models

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"{name}: AUC = {auc:.4f}")
```

## 📚 Documentation

- **[PARQUET_SETUP_GUIDE.md](PARQUET_SETUP_GUIDE.md)** - Complete setup and usage guide
- **[ENHANCED_vs_ORIGINAL.md](ENHANCED_vs_ORIGINAL.md)** - Detailed feature comparison
- **[GIT_COMMITS_PLAN.md](GIT_COMMITS_PLAN.md)** - Git workflow and commits

## 🐛 Troubleshooting

### Issue: "Input X contains infinity or a value too large"
**Solution**: Update `src/modeling/pipeline.py` to handle inf/-inf:
```python
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
```

### Issue: "File not found" for parquet files
**Solution**: Ensure paths in `src/modeling/main.py` match your directory structure:
```python
data_path="./data/final/nhanes_diabetes_engineered.parquet"
```

### Issue: SMOTE taking too long
**Normal behavior** for 54K+ samples. Expected: 5-10 seconds. Reduce data or `k_neighbors=3` for speed.

## 📊 Performance Metrics

The pipeline tracks:
- **Shape transformations** at each stage
- **Class distribution** (before/after SMOTE)
- **Missing value statistics**
- **Scaling parameters** (mean, std for each feature)
- **Execution time** for each phase

All saved in `prep_metadata.json` for reproducibility.

## 🔄 Workflow

```
Raw NHANES Data
    ↓
[Feature Engineering] - Creates 23 clinical features
    ↓
Engineered Data (46 features)
    ↓
[Data Cleaning] - Remove outliers, validate
    ↓
Clean Data (46 features, -2% rows)
    ↓
[Data Preparation] - Split, impute, SMOTE, scale
    ↓
ML-Ready Data
    ├── X_train (79,284 × 44) scaled
    ├── X_test (10,876 × 44) scaled
    ├── y_train (balanced)
    └── y_test (original dist.)
    ↓
[Model Training]
    ├── RandomForest
    ├── XGBoost
    ├── LogisticRegression
    └── ...
    ↓
[Evaluation]
    ├── ROC-AUC
    ├── Classification Report
    ├── Feature Importance
    └── Cross-validation
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License

Copyright (c) 2026 Pablo Monclús

## 🙏 Acknowledgments

- **NHANES Dataset**: CDC/NCHS (https://www.cdc.gov/nchs/nhanes/)
- **Feature Engineering**: Clinical guidelines and epidemiological research
- **SMOTE**: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique" (2002)
- **Friedewald Formula**: Friedewald et al., "Estimation of the Concentration of Low-Density Lipoprotein Cholesterol"

## 📞 Contact & Support

For questions, issues, or suggestions:
- Open an [Issue](https://github.com/vitwea/t2diabetes-predictor/issues)
- Start a [Discussion](https://github.com/vitwea/t2diabetes-predictor/discussions)

## 🎯 Roadmap

- [ ] Add cross-validation framework
- [ ] Implement hyperparameter tuning (Optuna/GridSearch)
- [ ] Add SHAP explainability
- [ ] Deploy as API (FastAPI)
- [ ] Add interpretability plots (feature importance, SHAP)
- [ ] Create interactive dashboard (Streamlit)
- [ ] Add model persistence (pickle/joblib)

## 📊 Dataset Citation

```bibtex
@misc{CDC2020,
  title={National Health and Nutrition Examination Survey (NHANES)},
  author={CDC/NCHS},
  year={2020},
  url={https://www.cdc.gov/nchs/nhanes/}
}
```

---

**Last Updated**: January 20, 2026  
**Status**: ✅ Active Development  
**Python**: 3.8+  
**Scikit-learn**: 1.0+
