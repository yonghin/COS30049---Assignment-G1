# =============================================================================
# FILE: 05_classification_models.py
# PURPOSE: Train and evaluate 3 classification models
#
# MODELS:
#   1. Random Forest  → Spam detection (combined SMS + Enron)
#   2. Naive Bayes    → Spam detection (TF-IDF features)
#   3. SVM            → Malware detection (MalMem features)
#
# WHY CLASSIFICATION?
#   Our core problem is labeling inputs as spam/ham or malware/benign.
#   Classification directly solves this — each model outputs a class label.
#   Using 3 different classifiers lets us compare which algorithm works best.
#
# OUTPUTS:
#   ../outputs/models/          → saved .pkl model files
#   ../outputs/visualizations/  → confusion matrices, comparison chart
#   ../outputs/classification_results.csv → metrics table
# =============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ── Create output folders ──────────────────────────────────────────────────
os.makedirs("../outputs/models", exist_ok=True)
os.makedirs("../outputs/visualizations", exist_ok=True)

# ── Shared helper functions ────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    """Print and return the 4 standard metrics required by the assignment."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n  {'─'*40}")
    print(f"  {name}")
    print(f"  {'─'*40}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1}

def save_confusion_matrix(y_true, y_pred, labels, title, fname):
    """Save a confusion matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'../outputs/visualizations/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {fname}")

all_results = []

# =============================================================================
# MODEL 1 — Random Forest: Spam Detection
# =============================================================================
print("\n" + "="*60)
print("MODEL 1: Random Forest — Spam Detection")
print("="*60)

try:
    df = pd.read_csv("../data/processed/combined_spam_processed.csv")
    print(f"✓ Loaded combined spam dataset: {df.shape}")

    feature_cols = ['message_length', 'word_count']
    feature_cols += [c for c in df.columns if c.startswith('has_')]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    y = df['label'].values
    print(f"  Features: {len(feature_cols)} | Spam: {y.sum()} | Ham: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 100 decision trees averaged together — robust and accurate
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    res = evaluate("Random Forest — Spam", y_test, y_pred)
    cv = cross_val_score(rf, X, y, cv=5, scoring='f1_weighted', n_jobs=-1)
    print(f"  5-Fold CV F1: {cv.mean():.4f} ± {cv.std():.4f}")
    res['CV_F1'] = round(cv.mean(), 4)
    all_results.append(res)

    # Feature importance chart
    importances = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    importances.plot(kind='barh', ax=ax, color='#2ecc71')
    ax.set_title('Random Forest — Top Feature Importances (Spam)', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('../outputs/visualizations/rf_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Feature importance chart saved")

    save_confusion_matrix(y_test, y_pred, ['Ham','Spam'],
                          'Random Forest — Spam Detection', 'cm_rf_spam.png')

    with open("../outputs/models/rf_spam.pkl", "wb") as f:
        pickle.dump({'model': rf, 'feature_cols': feature_cols}, f)
    print("  ✓ Model saved: rf_spam.pkl")

except FileNotFoundError as e:
    print(f"  ❌ File not found: {e}")

# =============================================================================
# MODEL 2 — Naive Bayes: Spam Detection (TF-IDF)
# =============================================================================
print("\n" + "="*60)
print("MODEL 2: Naive Bayes — Spam Detection (TF-IDF)")
print("="*60)

try:
    df_tfidf = pd.read_csv("../data/processed/sms_spam_tfidf.csv")
    print(f"✓ Loaded TF-IDF dataset: {df_tfidf.shape}")

    y_nb = df_tfidf['label_encoded'].values
    X_nb = df_tfidf.drop(columns=['label_encoded']).values

    # MultinomialNB needs non-negative values — scale to [0, 1]
    scaler = MinMaxScaler()
    X_nb = scaler.fit_transform(X_nb)

    X_train, X_test, y_train, y_test = train_test_split(
        X_nb, y_nb, test_size=0.2, random_state=42, stratify=y_nb)

    # alpha=1.0 = Laplace smoothing — prevents zero probability for unseen words
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    res = evaluate("Naive Bayes — Spam (TF-IDF)", y_test, y_pred)
    cv = cross_val_score(nb, X_nb, y_nb, cv=5, scoring='f1_weighted')
    print(f"  5-Fold CV F1: {cv.mean():.4f} ± {cv.std():.4f}")
    res['CV_F1'] = round(cv.mean(), 4)
    all_results.append(res)

    save_confusion_matrix(y_test, y_pred, ['Ham','Spam'],
                          'Naive Bayes — Spam (TF-IDF)', 'cm_nb_spam.png')

    with open("../outputs/models/nb_spam.pkl", "wb") as f:
        pickle.dump({'model': nb, 'scaler': scaler}, f)
    print("  ✓ Model saved: nb_spam.pkl")

except FileNotFoundError as e:
    print(f"  ❌ File not found: {e}")

# =============================================================================
# MODEL 3 — SVM: Malware Detection
# =============================================================================
print("\n" + "="*60)
print("MODEL 3: SVM — Malware Detection (MalMem)")
print("="*60)

try:
    df_mal = pd.read_csv("../data/processed/malmem_processed.csv")
    print(f"✓ Loaded MalMem dataset: {df_mal.shape}")

    drop_cols = [c for c in ['binary_label','category_encoded','category_name']
                 if c in df_mal.columns]
    X_svm = df_mal.drop(columns=drop_cols).values
    y_svm = df_mal['binary_label'].values

    # Sample 20k rows — SVM is O(n²) and very slow on 58k rows
    if len(X_svm) > 20000:
        idx = np.random.RandomState(42).choice(len(X_svm), 20000, replace=False)
        X_svm, y_svm = X_svm[idx], y_svm[idx]
        print(f"  Sampled 20,000 rows for training speed")

    X_train, X_test, y_train, y_test = train_test_split(
        X_svm, y_svm, test_size=0.2, random_state=42, stratify=y_svm)

    print("  Training SVM... (may take 2-3 minutes)")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    res = evaluate("SVM — Malware Detection", y_test, y_pred)
    res['CV_F1'] = 'N/A'
    all_results.append(res)

    save_confusion_matrix(y_test, y_pred, ['Benign','Malware'],
                          'SVM — Malware Detection', 'cm_svm_malware.png')

    with open("../outputs/models/svm_malware.pkl", "wb") as f:
        pickle.dump(svm, f)
    print("  ✓ Model saved: svm_malware.pkl")

except FileNotFoundError as e:
    print(f"  ❌ File not found: {e}")

# =============================================================================
# FINAL SUMMARY + comparison chart
# =============================================================================
print("\n" + "="*60)
print("CLASSIFICATION RESULTS SUMMARY")
print("="*60)

if all_results:
    results_df = pd.DataFrame(all_results)
    print(results_df[['Model','Accuracy','Precision','Recall','F1']].to_string(index=False))
    results_df.to_csv("../outputs/classification_results.csv", index=False)
    print("\n✓ Saved: ../outputs/classification_results.csv")

    # Grouped bar chart
    metrics = ['Accuracy','Precision','Recall','F1']
    colors  = ['#3498db','#2ecc71','#e67e22','#e74c3c']
    x = np.arange(len(results_df))
    w = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w, pd.to_numeric(results_df[m], errors='coerce'), w, label=m, color=c)

    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(results_df['Model'], fontsize=8, rotation=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Classification Models — Performance Comparison',
                 fontsize=13, fontweight='bold')
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig('../outputs/visualizations/classification_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved comparison chart: classification_comparison.png")

print("\n✅ Classification done! Next: python 06_clustering_models.py")
