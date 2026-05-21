# 🛡️ NTCyber AI — Spam & Malware Detection System

> **COS30049 Computing Technology Innovation Project — Assignment 2**
> Session 1 | Group 1 | Section C1

A machine learning system that detects spam messages and malware threats using classification, clustering, and regression models trained on four real-world cybersecurity datasets.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Datasets](#datasets)
4. [Environment Setup](#environment-setup)
5. [Step 1 — Data Preprocessing](#step-1--data-preprocessing)
6. [Step 2 — Model Training](#step-2--model-training)
7. [Step 3 — Model Validation & Charts](#step-3--model-validation--charts)
8. [Step 4 — Making Predictions](#step-4--making-predictions)
9. [Model Summary](#model-summary)
10. [Results](#results)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

NTCyber AI implements six machine learning models across three method types:

| Method Type    | Models                          | Task                              |
| -------------- | ------------------------------- | --------------------------------- |
| Classification | Random Forest, Naive Bayes, SVM | Spam detection, Malware detection |
| Clustering     | K-Means, DBSCAN                 | Malware family grouping           |
| Regression     | Logistic Regression             | Spam probability scoring          |

**Tech Stack:** Python 3.10 · scikit-learn · pandas · matplotlib · seaborn

---

## Project Structure

```
COS30049---Assignment-G1/
├── data/
│   └── processed/          ← Auto-generated after preprocessing
│       ├── category_mapping.json
│       ├── emails_inti_processed.csv
│       ├── enron_processed.csv
│       ├── malmem_processed.csv
│       ├── malmem_scaler.pkl
│       ├── malware_basic_processed.csv
│       ├── sms_spam_processed.csv
│       ├── sms_spam_tfidf.csv
│    └── raw/                 ← Place downloaded datasets here
│       ├── emails_inti.csv
│       ├── enron_spam_data.csv
│       ├── Malware_dataset.csv
│       ├── Obfuscated-MalMem2022.csv
│       └── SMSSpamCollection
├── models/
│   ├── 05_classification_models.py   ← Random Forest, Naive Bayes, SVM
│   ├── 06_clustering_models.py       ← K-Means, DBSCAN
│   ├── 07_regression_model.py        ← Logistic Regression
│   ├── 08_run_all_models.py          ← Master script (runs all models)
│   ├── 09_validation_and_insights.py
│   └── 10_fix_and_enhance_charts.py
├── outputs/
│   └── models/              ← Saved .pkl model files
│       ├── dbscan_malware.pkl
│       ├── kmeans_malware.pkl
│       ├── logistic_regression_spam.pkl
│       ├── nb_spam.pkl
│       ├── rf_spam.pkl
│       └──  svm_malware.pkl
│   └── validation/          ← CSV results tables
│       ├── cross_validation_results.csv
│       ├── error_analysis.csv
│       ├── live_predictions.csv
│       └── model_ranking.csv
│   └── visualizations/      ← All generated charts (PNG)
│       ├── classification_comparison.png
│       ├── cluster_size_comparison.png
│       ├── cm_nb_spam.png
│       ├── cm_rf_spam.png
│       ├── cm_svm_malware.png
│       ├── cross_validation_comparison.png
│       ├── dbscan_clusters_fixed.png
│       ├── dbscan_clusters.png
│       ├── emails_inti_analysis.png
│       ├── enron_analysis.png
│       ├── error_analysis.png
│       ├── final_model_ranking.png
│       ├── kmeans_clusters.png
│       ├── kmeans_elbow.png
│       ├── lc_lr_spam.png
│       ├── lc_nb_spam.png
│       ├── lc_rf_spam.png
│       ├── lc_svm_malware.png
│       ├── lr_coefficients.png
│       ├── lr_confusion_matrix.png
│       ├── lr_probability_distribution.png
│       ├── lr_roc_curve.png
│       ├── malmem_analysis_fixed.png
│       ├── malmem_correlation_heatmap.png
│       ├── malmem_feature_boxplot.png
│       ├── malware_basic_analysis.png
│       ├── pca_scree_plot.png
│       ├── rf_feature_importance.png
│       ├── roc_all_models.png
│       ├── sms_spam_analysis.png
│       ├── spam_keyword_heatmap.png
│       ├── all_results_combined.csv
│       ├── classification_results.csv
│       ├── clustering_results.csv
│       └── regression_results.csv
├── preprocessing/
│   ├── 00_run_all_preprocessing.py   ← Master script (runs all preprocessing)
│   ├── 01_preprocess_sms_spam.py
│   ├── 02_preprocess_malmem.py
│   ├── 03_preprocess_enron.py
│   ├── 04_preprocess_basic_datasets.py
│   └── README_preprocessing.md
├── .gitignore
└── README.md

```

---

## Datasets

Download all four datasets and place them in the `data/raw/` folder **before running any scripts**.

| #   | Dataset               | Source                                                                                      | Filename to use                          |
| --- | --------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 1   | SMS Spam Collection   | [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)            | `SMSSpamCollection`                      |
| 2   | Enron Email Spam Data | [Kaggle — marcelwiechmann](https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data) | `enron_spam_data.csv`                    |
| 3   | CIC-MalMem-2022       | [Kaggle — jlcole](https://www.kaggle.com/datasets/jlcole/cic-malmem-2022)                   | `Obfuscated-MalMem2022.csv`              |
| 4   | Unit Basic Datasets   | Provided by course                                                                          | `emails_inti.csv`, `Malware_dataset.csv` |

> ⚠️ **Important:** For the Enron dataset, use the **marcelwiechmann** version (33,716 labelled emails).
> Do NOT use the wcukierski version — it is unlabelled and incompatible with this pipeline.

> ⚠️ **Excel files:** If your downloaded files are `.xlsx`, open them in Excel and
> **Save As → CSV (Comma delimited)** before placing them in `data/raw/`.

Your `data/raw/` folder should contain:

```
data/raw/
├── SMSSpamCollection           (no extension)
├── enron_spam_data.csv
├── Obfuscated-MalMem2022.csv
├── emails_inti.csv
└── Malware_dataset.csv
```

---

## Environment Setup

### Prerequisites

- [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) or Anaconda installed
- Python 3.10
- ~3 GB free disk space (for datasets and processed files)

### 1. Create and activate the conda environment

Open **Anaconda Prompt** (not Git Bash) and run:

```bash
conda create -n spam_malware python=3.10
```

When prompted `Proceed ([y]/n)?` type `y` and press Enter.

```bash
conda activate spam_malware
```

Your prompt should now show `(spam_malware)` at the start.

### 2. Install required packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

This installs all dependencies needed for preprocessing, training, and evaluation.

### 3. Verify installation

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages OK')"
```

Expected output:

```
All packages OK
```

### 4. Navigate to the project folder

```bash
cd "C:\path\to\your\COS30049---Assignment-G1"
```

> Replace the path above with the actual path to your project folder.

---

## Step 1 — Data Preprocessing

Preprocessing cleans and prepares all four datasets for model training.
Run from the `preprocessing/` folder:

```bash
cd preprocessing
python 00_run_all_preprocessing.py
```

This master script automatically runs all four preprocessing scripts in order:

| Script                            | Dataset             | What it does                                                               |
| --------------------------------- | ------------------- | -------------------------------------------------------------------------- |
| `01_preprocess_sms_spam.py`       | SMS Spam Collection | Text cleaning, TF-IDF vectorisation, feature engineering                   |
| `02_preprocess_malmem.py`         | CIC-MalMem-2022     | Missing value imputation, variance filtering, StandardScaler normalisation |
| `03_preprocess_enron.py`          | Enron Email         | Text cleaning, TF-IDF vectorisation, label standardisation                 |
| `04_preprocess_basic_datasets.py` | Unit basic datasets | Auto-detection of columns, cleaning, scaling                               |

### Expected output files in `data/processed/`

```
data/processed/
├── sms_spam_processed.csv        ← SMS spam features (engineered)
├── sms_spam_tfidf.csv            ← SMS spam TF-IDF features (500 words)
├── malmem_processed.csv          ← MalMem scaled features (39 features)
├── malmem_scaler.pkl             ← Saved StandardScaler for predictions
├── category_mapping.json         ← Malware category label mapping
├── enron_processed.csv           ← Enron email features
├── combined_spam_processed.csv   ← SMS + Enron merged (for Random Forest)
├── emails_inti_processed.csv     ← Unit basic email dataset
└── malware_basic_processed.csv   ← Unit basic malware dataset
```

### Expected console output (summary)

```
✅ Success  SMS Spam Collection (UCI)
✅ Success  CIC-MalMem-2022 (Malware)
✅ Success  Enron Email Dataset
✅ Success  Combined Spam Dataset
```

> ℹ️ Preprocessing generates visualisation charts saved to `outputs/visualizations/`.
> Charts save silently to PNG files — no popup windows will appear.

---

## Step 2 — Model Training

Train all six models by running the master model script from the `models/` folder:

```bash
cd ../models
python 08_run_all_models.py
```

This runs three training scripts in order:

### Script 1 — Classification Models (`05_classification_models.py`)

Trains three classifiers:

| Model         | Dataset                             | Parameters                                            |
| ------------- | ----------------------------------- | ----------------------------------------------------- |
| Random Forest | Combined spam (engineered features) | `n_estimators=100`, `max_depth=15`, `random_state=42` |
| Naive Bayes   | SMS spam (TF-IDF)                   | `alpha=1.0` (Laplace smoothing)                       |
| SVM           | MalMem malware features             | `kernel='rbf'`, `C=1.0`, `gamma='scale'`              |

> ⏱️ **SVM training takes approximately 2–3 minutes** on a 20,000-sample subset. This is normal.

### Script 2 — Clustering Models (`06_clustering_models.py`)

Trains two clustering models:

| Model   | Dataset                    | Parameters                                              |
| ------- | -------------------------- | ------------------------------------------------------- |
| K-Means | MalMem (10 PCA components) | `n_clusters` auto-detected, `n_init=10`, `max_iter=300` |
| DBSCAN  | MalMem (5 PCA components)  | `eps=0.8`, `min_samples=15`                             |

### Script 3 — Regression Model (`07_regression_model.py`)

Trains Logistic Regression on SMS spam TF-IDF features:

| Model               | Dataset           | Parameters                                 |
| ------------------- | ----------------- | ------------------------------------------ |
| Logistic Regression | SMS spam (TF-IDF) | `C=1.0`, `max_iter=1000`, `solver='lbfgs'` |

### Expected saved model files in `outputs/models/`

```
outputs/models/
├── rf_spam.pkl                      ← Random Forest spam classifier
├── nb_spam.pkl                      ← Naive Bayes spam classifier
├── svm_malware.pkl                  ← SVM malware classifier
├── kmeans_malware.pkl               ← K-Means malware clustering model
├── dbscan_malware.pkl               ← DBSCAN anomaly detection model
└── logistic_regression_spam.pkl     ← Logistic Regression spam scorer
```

### Expected console summary at the end

```
✅ Success  Classification  (RF, Naive Bayes, SVM)
✅ Success  Clustering      (K-Means, DBSCAN)
✅ Success  Regression      (Logistic Regression)
```

---

## Step 3 — Model Validation & Charts

Run validation and generate all report-quality charts:

```bash
# Validation: learning curves, cross-validation, ROC curves, error analysis
python 09_validation_and_insights.py

# Fix charts and generate additional visualisations (heatmaps, boxplots, etc.)
python 10_fix_and_enhance_charts.py
```

### What `09_validation_and_insights.py` generates

| Output                           | Location                                          |
| -------------------------------- | ------------------------------------------------- |
| Learning curve charts (4 models) | `outputs/visualizations/lc_*.png`                 |
| Cross-validation results table   | `outputs/validation/cross_validation_results.csv` |
| ROC curves (all models)          | `outputs/visualizations/roc_all_models.png`       |
| Error analysis (FP vs FN)        | `outputs/visualizations/error_analysis.png`       |
| Live prediction test results     | `outputs/validation/live_predictions.csv`         |
| Final model ranking table        | `outputs/validation/model_ranking.csv`            |

### What `10_fix_and_enhance_charts.py` generates

| Chart                            | Description                                   |
| -------------------------------- | --------------------------------------------- |
| `dbscan_clusters_fixed.png`      | DBSCAN scatter plot (dark background)         |
| `malmem_analysis_fixed.png`      | MalMem category distribution (fixed labels)   |
| `malmem_correlation_heatmap.png` | Feature correlation heatmap (top 12 features) |
| `malmem_feature_boxplot.png`     | Feature distributions by malware category     |
| `pca_scree_plot.png`             | PCA explained variance scree plot             |
| `spam_keyword_heatmap.png`       | Spam keyword frequency heatmap                |
| `cluster_size_comparison.png`    | K-Means vs DBSCAN cluster sizes               |

> ⏱️ `09_validation_and_insights.py` takes approximately 5–10 minutes to run
> because it computes learning curves using cross-validation on all models.

---

## Step 4 — Making Predictions

Use the trained models to classify new messages or memory samples.

### Predict spam probability for a new message

```python
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load the Logistic Regression model ---
with open("outputs/models/logistic_regression_spam.pkl", "rb") as f:
    lr_data = pickle.load(f)

model   = lr_data['model']
scaler  = lr_data['scaler']
feature_names = lr_data['feature_names']

# --- Load and refit the TF-IDF vectorizer ---
df = pd.read_csv("data/processed/sms_spam_processed.csv")

tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
tfidf.fit(df['cleaned_message'])

# --- Clean and predict a new message ---
def clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

message = "Congratulations! You've won a FREE iPhone. Click here to claim your prize!"
cleaned = clean(message)

vec        = tfidf.transform([cleaned]).toarray()
vec_scaled = scaler.transform(vec)

prediction   = model.predict(vec_scaled)[0]
spam_prob    = model.predict_proba(vec_scaled)[0][1]
label        = "SPAM" if prediction == 1 else "HAM"

print(f"Message:      {message}")
print(f"Prediction:   {label}")
print(f"Spam Prob:    {spam_prob:.4f} ({spam_prob*100:.1f}%)")
```

**Example output:**

```
Message:      Congratulations! You've won a FREE iPhone. Click here to claim your prize!
Prediction:   SPAM
Spam Prob:    0.9821 (98.2%)
```

---

### Classify a memory sample as malware or benign

```python
import pickle
import numpy as np
import pandas as pd

# --- Load the SVM model and scaler ---
with open("outputs/models/svm_malware.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("data/processed/malmem_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Load feature column names ---
df = pd.read_csv("data/processed/malmem_processed.csv")
drop_cols = [c for c in ['binary_label', 'category_encoded', 'category_name'] if c in df.columns]
feature_cols = [c for c in df.columns if c not in drop_cols]

# --- Predict on a sample from the dataset ---
# (Replace with real memory feature values for actual use)
sample = df[feature_cols].iloc[0:1].values   # First sample as example

prediction = svm_model.predict(sample)[0]
probability = svm_model.predict_proba(sample)[0]

label = "MALWARE" if prediction == 1 else "BENIGN"
print(f"Prediction:       {label}")
print(f"Benign Prob:      {probability[0]:.4f}")
print(f"Malware Prob:     {probability[1]:.4f}")
```

**Example output:**

```
Prediction:       BENIGN
Benign Prob:      0.9934
Malware Prob:     0.0066
```

---

### Run the full prediction pipeline (spam + malware)

Save the following as `predict.py` in the project root and run it:

```bash
python predict.py
```

```python
# predict.py — Quick demo of all trained models
import pickle, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 55)
print("  NTCyber AI — Prediction Demo")
print("=" * 55)

# ── Spam prediction (Naive Bayes) ──────────────────────
with open("outputs/models/nb_spam.pkl", "rb") as f:
    nb = pickle.load(f)

df_sms = pd.read_csv("data/processed/sms_spam_processed.csv")
tfidf  = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))
tfidf.fit(df_sms['cleaned_message'])

def clean(t):
    t = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', str(t).lower())
    return re.sub(r'\s+', ' ', t).strip()

test_messages = [
    "Win a FREE iPhone now! Click here to claim your prize!",
    "Hey, are you coming to lunch today at 1pm?",
    "URGENT: Your account will be suspended. Verify now.",
]

print("\n📧 SPAM DETECTION (Naive Bayes)")
print(f"  {'Message':<48} {'Result':>8} {'Confidence':>12}")
print(f"  {'─'*48} {'─'*8} {'─'*12}")
for msg in test_messages:
    vec   = nb['scaler'].transform(tfidf.transform([clean(msg)]).toarray())
    pred  = nb['model'].predict(vec)[0]
    prob  = nb['model'].predict_proba(vec)[0][1]
    label = "SPAM ⚠️" if pred == 1 else "HAM  ✓"
    short = (msg[:45] + "...") if len(msg) > 48 else msg
    print(f"  {short:<48} {label:>8} {prob*100:>11.1f}%")

# ── Malware prediction (SVM) ───────────────────────────
with open("outputs/models/svm_malware.pkl", "rb") as f:
    svm = pickle.load(f)

df_mal   = pd.read_csv("data/processed/malmem_processed.csv")
drop     = [c for c in ['binary_label','category_encoded','category_name'] if c in df_mal.columns]
feat     = [c for c in df_mal.columns if c not in drop]
samples  = df_mal[feat].iloc[:3].values
true_lbl = df_mal['binary_label'].iloc[:3].values

print("\n🦠 MALWARE DETECTION (SVM)")
print(f"  {'Sample':<10} {'Prediction':>12} {'Confidence':>12} {'Actual':>10}")
print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*10}")
for i, (s, t) in enumerate(zip(samples, true_lbl)):
    pred = svm.predict([s])[0]
    prob = svm.predict_proba([s])[0][1]
    plbl = "MALWARE ⚠️" if pred == 1 else "BENIGN  ✓"
    albl = "Malware" if t == 1 else "Benign"
    print(f"  Sample {i+1:<3} {plbl:>12} {prob*100:>11.1f}% {albl:>10}")

print("\n✅ Prediction demo complete.")
```

---

## Model Summary

| Model               | Type           | Task              | Accuracy | F1 Score             | AUC-ROC |
| ------------------- | -------------- | ----------------- | -------- | -------------------- | ------- |
| SVM                 | Classification | Malware detection | 99.92%   | 0.9993               | 1.0000  |
| Random Forest       | Classification | Spam detection    | 98.39%   | 0.9839               | 0.9978  |
| Naive Bayes         | Classification | Spam detection    | 96.71%   | 0.9662               | 0.9787  |
| Logistic Regression | Regression     | Spam probability  | 96.13%   | 0.8276               | 0.9899  |
| K-Means             | Clustering     | Malware grouping  | —        | Silhouette: 0.5668   | —       |
| DBSCAN              | Clustering     | Anomaly detection | —        | 710 anomalies (4.7%) | —       |

---

## Results

All results, charts, and validation files are saved automatically after running the scripts:

```
outputs/
├── models/                          ← 6 trained model .pkl files
├── visualizations/                  ← 20+ PNG charts for the report
├── validation/
│   ├── cross_validation_results.csv
│   ├── error_analysis.csv
│   ├── live_predictions.csv
│   └── model_ranking.csv
├── classification_results.csv
├── clustering_results.csv
├── regression_results.csv
└── all_results_combined.csv
```

---

## Troubleshooting

| Problem                                  | Cause                                           | Fix                                                                |
| ---------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------ |
| `conda: command not found`               | Running in Git Bash instead of Anaconda Prompt  | Open **Anaconda Prompt** from the Start Menu                       |
| `ModuleNotFoundError: sklearn`           | Running in `(base)` instead of `(spam_malware)` | Run `conda activate spam_malware` first                            |
| `FileNotFoundError: enron_spam_data.csv` | Wrong Enron version downloaded                  | Download from **marcelwiechmann** on Kaggle                        |
| `ValueError: n_clusters=28346`           | Wrong label column used for K-Means             | Fixed in `06_clustering_models.py` — re-download latest version    |
| Script pauses at Step 8 (charts)         | `plt.show()` blocking the script                | Fixed in latest scripts — uses `Agg` backend + `plt.close()`       |
| SVM takes very long                      | Training on full 58k rows                       | Normal — script samples 20,000 rows automatically                  |
| `KeyboardInterrupt` on heatmap           | Seaborn rendering too slow                      | Fixed in latest `02_preprocess_malmem.py` — uses bar chart instead |

---

## Team

| Name         | Student ID | Role                                         |
| ------------ | ---------- | -------------------------------------------- |
| Tee Ren Hang | 106214467  | Project Manager, Report Lead, UI/UX Designer |
| Ng Yong Hin  | 106214441  | Technical Lead, ML Implementation            |

**Lecturer:** Mr. Faizal | **Section:** C1 | **Unit:** COS30049
