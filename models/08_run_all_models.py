# =============================================================================
# FILE: 08_run_all_models.py
# PURPOSE: Master script — runs all 3 model scripts in order and
#          prints a final combined summary of every model's performance.
#
# HOW TO USE:
#   cd models
#   python 08_run_all_models.py
# =============================================================================

import subprocess
import sys
import os
import pandas as pd

print("=" * 60)
print("  SPAM & MALWARE DETECTION — MODEL TRAINING PIPELINE")
print("=" * 60)
print("Running 3 model scripts in order...\n")

scripts = [
    ("05_classification_models.py", "Classification  (RF, Naive Bayes, SVM)"),
    ("06_clustering_models.py",     "Clustering      (K-Means, DBSCAN)"),
    ("07_regression_model.py",      "Regression      (Logistic Regression)"),
]

statuses = {}

for script, label in scripts:
    print(f"\n{'='*60}")
    print(f"▶  {label}")
    print(f"   Script: {script}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    statuses[label] = "✅ Success" if result.returncode == 0 else "❌ Failed"

# =============================================================================
# Combined summary across all output CSVs
# =============================================================================
print(f"\n\n{'='*60}")
print("  FINAL RESULTS — ALL MODELS")
print(f"{'='*60}")

all_rows = []

for fname in ["../outputs/classification_results.csv",
              "../outputs/clustering_results.csv",
              "../outputs/regression_results.csv"]:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        all_rows.append(df)

if all_rows:
    combined = pd.concat(all_rows, ignore_index=True)
    cols = [c for c in ['Model','Accuracy','Precision','Recall','F1','AUC_ROC',
                         'Silhouette','N_Clusters'] if c in combined.columns]
    print(combined[cols].to_string(index=False))
    combined.to_csv("../outputs/all_results_combined.csv", index=False)
    print("\n✓ Saved: ../outputs/all_results_combined.csv")

print(f"\n{'='*60}")
print("  SCRIPT STATUS SUMMARY")
print(f"{'='*60}")
for label, status in statuses.items():
    print(f"  {status}  {label}")

print(f"\n{'='*60}")
print("  OUTPUT FILES")
print(f"{'='*60}")

output_files = [
    ("../outputs/models/rf_spam.pkl",                    "Random Forest — Spam"),
    ("../outputs/models/nb_spam.pkl",                    "Naive Bayes — Spam"),
    ("../outputs/models/svm_malware.pkl",                "SVM — Malware"),
    ("../outputs/models/kmeans_malware.pkl",             "K-Means — Malware"),
    ("../outputs/models/dbscan_malware.pkl",             "DBSCAN — Malware"),
    ("../outputs/models/logistic_regression_spam.pkl",   "Logistic Regression — Spam"),
    ("../outputs/classification_results.csv",            "Classification metrics"),
    ("../outputs/clustering_results.csv",                "Clustering metrics"),
    ("../outputs/regression_results.csv",                "Regression metrics"),
    ("../outputs/all_results_combined.csv",              "All results combined"),
]

for fpath, desc in output_files:
    exists = "✓" if os.path.exists(fpath) else "✗"
    size   = f"({os.path.getsize(fpath)/1024:.0f} KB)" if os.path.exists(fpath) else ""
    print(f"  {exists} {desc:<35} {size}")

print(f"\n{'='*60}")
print("  VISUALISATIONS SAVED")
print(f"{'='*60}")

viz_dir = "../outputs/visualizations"
if os.path.exists(viz_dir):
    for f in sorted(os.listdir(viz_dir)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(viz_dir, f)) / 1024
            print(f"  ✓ {f:<50} ({size:.0f} KB)")

print("\n🎉 All models trained and evaluated!")
print("   Use the saved .pkl files in your web app for predictions.")
