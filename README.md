# 🩺 t2diabetes-predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC-0.854-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-NHANES%20CDC-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  <b>Binary classification model to predict Type 2 Diabetes risk using clinical, anthropometric and socioeconomic features from the NHANES dataset, with full SHAP-based explainability.</b>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Model](#-model)
- [Results](#-results)
- [Explainability (SHAP)](#-explainability-shap)
- [Streamlit Web App](#-streamlit-web-app)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [License](#-license)

---

## 🔍 Overview

Type 2 diabetes affects over 400 million people worldwide and remains largely underdiagnosed. This project builds a **production-ready binary classifier** using `HistGradientBoostingClassifier` to identify individuals at high risk of T2 diabetes based on routine health measurements.

Key highlights:

- **Threshold optimization** at 0.310 to maximize recall for the positive class (diabetic), prioritizing sensitivity in a clinical context
- **SHAP explainability** to interpret every prediction and understand global feature importance
- **Engineered interaction features** (e.g., `age_bmi_interaction`, `waist_height_ratio`) that significantly boost predictive power
- Trained on **28,452 samples** from the nationally representative NHANES survey

---

## 📊 Dataset

**Source:** [NHANES – National Health and Nutrition Examination Survey](https://www.cdc.gov/nchs/nhanes/index.htm) (CDC)

NHANES is a cross-sectional, nationally representative survey of the U.S. civilian non-institutionalized population. It combines interview and physical examination data, making it ideal for metabolic disease modeling.

| Split | Samples |
|-------|---------|
| Class 0 (No diabetes) | 18,260 |
| Class 1 (Diabetes) | 10,192 |
| **Total** | **28,452** |

> ⚠️ The dataset presents moderate class imbalance (~64/36), which is addressed via threshold tuning rather than resampling, preserving the original data distribution.

---

## ⚙️ Feature Engineering

Raw NHANES variables were enriched with clinically meaningful derived features:

| Feature | Description |
|---------|-------------|
| `age_bmi_interaction` | Age × BMI interaction term — top predictor |
| `waist_height_ratio` | Waist circumference / height (stronger than BMI alone for visceral fat) |
| `triglyceride_ratio` | Triglycerides relative ratio |
| `cholesterol_ratio` | Total cholesterol / HDL ratio |
| `age_group` | Discretized age buckets |

The final feature set includes 20 variables spanning **anthropometrics**, **blood markers**, **blood pressure**, **lifestyle** (sleep hours), and **socioeconomic** (income/poverty ratio) domains.

---

## 🤖 Model

**Algorithm:** `HistGradientBoostingClassifier` (scikit-learn)

Chosen for its:
- Native handling of missing values (no imputation needed)
- Excellent performance on tabular health data
- Fast training via histogram-based binning
- Compatibility with SHAP TreeExplainer

**Threshold:** 0.310 (optimized for high recall on class 1)

```python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    # hyperparameters tuned via cross-validation
)
```

---

## 📈 Results

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 – No Diabetes | 0.90 | 0.69 | 0.78 | 18,260 |
| 1 – Diabetes | 0.61 | 0.86 | 0.71 | 10,192 |
| **Weighted Avg** | **0.79** | **0.75** | **0.75** | 28,452 |

### Key Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.854** |
| Average Precision (AP) | **0.755** |
| Accuracy | 0.75 |
| Recall (Diabetes class) | **0.86** |

### Evaluation Plots

![Model Evaluation](reports/model_evaluation.png)

> The calibration curve confirms the model produces well-calibrated probability estimates, making the predicted scores directly interpretable as risk probabilities.

---

## 🔎 Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values are computed for every prediction, providing both global and local interpretability.

### Global Feature Importance

![SHAP Global Importance](reports/SHAP_global_importance.png)

### SHAP Summary Plot

![SHAP Summary](reports/SHAP_summary_plot.png)

**Key findings from SHAP analysis:**

- **`age_bmi_interaction`** is by far the most impactful feature — high values strongly increase predicted risk
- **`ethnicity`** is the second most important feature, reflecting known epidemiological disparities in T2 diabetes prevalence
- **`age_years`** reinforces the role of age as a primary risk factor
- **`waist_height_ratio`** outperforms raw BMI as a proxy for central adiposity
- **`hdl_cholesterol`** shows a protective effect — low HDL increases risk
- **`income_poverty_ratio`** highlights the socioeconomic dimension of metabolic disease

---

## 🌐 Streamlit Web App

The project includes an interactive web interface built with **Streamlit** (`streamlit_app.py`) that wraps the prediction pipeline from `app.py` into a user-friendly clinical dashboard.

### Features

- **Sidebar input form** — all 15 raw clinical variables grouped by category: Anthropometric, Blood Markers, Blood Pressure, and Lifestyle & Socioeconomic
- **Risk probability gauge** — animated Plotly gauge displaying the predicted probability with color-coded risk zones (Low / Moderate / High / Very High)
- **Risk banner** — instant visual summary with contextual clinical guidance for each risk level
- **Key metrics panel** — shows probability %, risk level, binary prediction, and the decision threshold in use
- **Per-patient SHAP chart** — interactive horizontal bar chart showing the top 10 features pushing risk up (red) or down (green) for the specific input
- **Full SHAP table** — expandable table with SHAP values and patient values for all features
- **Dark UI theme** — custom CSS with a dark navy palette and teal accent color

### Input Variables

| Category | Fields |
|----------|--------|
| ⚖️ Anthropometric | Age, Height, Weight, Waist circumference, BMI (auto-calculated) |
| 🩸 Blood Markers | HDL cholesterol, Total cholesterol, Triglycerides, Creatinine |
| 💓 Blood Pressure | Systolic BP, Diastolic BP, Hypertension diagnosis |
| 🌿 Lifestyle & Socioeconomic | Sleep hours/night, Income/Poverty ratio, Ethnicity |

### Risk Level Thresholds

| Level | Probability Range | Interpretation |
|-------|-------------------|----------------|
| ✅ Low | < 25% | Maintain a healthy lifestyle |
| ⚠️ Moderate | 25% – 50% | Consider preventive screening |
| 🔶 High | 50% – 70% | Clinical evaluation recommended |
| 🔴 Very High | ≥ 70% | Seek medical attention promptly |

> The decision threshold (31%) is shown as a reference line on the gauge. SHAP explainability requires `models/shap_explainer.pkl` — generate it by running `python generate_shap_explainer.py`.

### Run the app

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501` by default.

---

## 🗂️ Project Structure

```
t2diabetes-predictor/
│
├── data/
│   ├── dataset/		# Processed test/train parquet and artifacts             
│   ├── models/           	# Trained models created in notebooks
│   └── nhanes_data		# Datasets raw and clean
│
├── notebooks/
│   └──/...
│
├── src/
│   ├── features/ 	     	# Feature engineering, selection and clinical rules
│   ├── train/               	# Model training and evaluation
│   ├── app/                   	# API or application layer for serving the trained model
│   ├── pipeline/              	# End‑to‑end data pipelines: loading, validation, cleaning, imputation, scaling
│   ├── predict/		# Inference utilities: load model, apply threshold, generate predictions
│   ├── preprocessing/		# General preprocessing: NaN handling, outlier removal, encoding, normalization
│   ├── utils/			# Shared utilities: logging, configuration, metrics, helper functions
│   └── data/			# Data access layer: dataset loaders, paths, and I/O helpers
│
├── reports/
│   └── model_evaluation.png
│
├── model/
│   └── final_diabetes_model.pkl	# Serialized trained model
│
├── requirements.txt
├── streamlit_app.py          	# Streamlit web UI
└── README.md
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/vitwea/t2diabetes-predictor.git
cd t2diabetes-predictor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Main dependencies:**

```
scikit-learn>=1.3
shap>=0.44
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.32
plotly>=5.18
```

---

## 🚀 Usage


### Run the web app

```bash
streamlit run streamlit_app.py
```

The interactive dashboard opens at `http://localhost:8501`. Fill in the patient data in the sidebar and click **Predict Risk** to get the full risk assessment with SHAP explainability.

### CLI — Run predictions from file

### Run predictions from Python

```python
import pickle
import pandas as pd

# Load model
with open("model/t2diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict (returns probability of diabetes)
sample = pd.DataFrame([{
    "age_years": 55,
    "bmi": 31.2,
    "age_bmi_interaction": 55 * 31.2,
    "waist_height_ratio": 0.61,
    # ... other features
}])

risk_prob = model.predict_proba(sample)[:, 1]
prediction = (risk_prob >= 0.310).astype(int)

print(f"Diabetes risk probability: {risk_prob[0]:.2%}")
print(f"Predicted class: {'High Risk' if prediction[0] else 'Low Risk'}")
```

---

## ⚠️ Disclaimer

This model is intended for **research and educational purposes only**. It is not a validated clinical diagnostic tool and should not be used as a substitute for professional medical evaluation. Predictions are probabilistic estimates based on population-level data.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  Made with ❤️ and data from the CDC NHANES survey
</p>
