# =============================================================================
# FILE: 07_regression_model.py
# PURPOSE: Train a Logistic Regression model for spam probability scoring
#
# MODEL:
#   Logistic Regression → Predicts the PROBABILITY that a message is spam
#
# WHY LOGISTIC REGRESSION AS THE "REGRESSION" METHOD?
#   The assignment requires at least one regression-type method.
#   Logistic Regression is a regression model (it fits a regression line) but
#   outputs a probability between 0.0 and 1.0 — making it ideal for scoring
#   how "spammy" a message is rather than just binary spam/not-spam.
#   This is how real-world email spam scores work (e.g. SpamAssassin scores).
#   It also provides interpretable coefficients showing which words/features
#   drive spam classification — useful for the report.
#
# OUTPUTS:
#   ../outputs/models/logistic_regression_spam.pkl
#   ../outputs/visualizations/lr_coefficients.png
#   ../outputs/visualizations/lr_probability_distribution.png
#   ../outputs/regression_results.csv
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)

os.makedirs("../outputs/models", exist_ok=True)
os.makedirs("../outputs/visualizations", exist_ok=True)

# =============================================================================
# Load TF-IDF spam dataset
# Logistic Regression works well with TF-IDF features because each word's
# coefficient tells us directly how much it contributes to spam prediction
# =============================================================================
print("="*60)
print("MODEL: Logistic Regression — Spam Probability Scoring")
print("="*60)

try:
    df = pd.read_csv("../data/processed/sms_spam_tfidf.csv")
    print(f"✓ Loaded TF-IDF dataset: {df.shape}")
except FileNotFoundError:
    print("❌ sms_spam_tfidf.csv not found. Run preprocessing first.")
    exit()

y = df['label_encoded'].values
X = df.drop(columns=['label_encoded']).values
feature_names = df.drop(columns=['label_encoded']).columns.tolist()

print(f"  Spam: {y.sum()}  Ham: {(y==0).sum()}")
print(f"  Features (TF-IDF words): {X.shape[1]}")

# Scale to [0, 1] — required for LogisticRegression stability
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 80/20 train-test split, stratified to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

# =============================================================================
# Train Logistic Regression
# max_iter=1000 ensures convergence on high-dimensional TF-IDF data
# C=1.0 = regularization strength (smaller C = stronger regularization)
# solver='lbfgs' = efficient for multi-dimensional problems
# =============================================================================
print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
lr.fit(X_train, y_train)

y_pred      = lr.predict(X_test)
y_prob      = lr.predict_proba(X_test)[:, 1]   # Probability of spam (class 1)

print("✓ Training complete!")

# =============================================================================
# Evaluation metrics
# =============================================================================
print("\n" + "─"*50)
print("  LOGISTIC REGRESSION — EVALUATION METRICS")
print("─"*50)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)   # AUC-ROC: 1.0 = perfect, 0.5 = random

print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  AUC-ROC  : {auc:.4f}  ← regression-specific metric (area under ROC curve)")
print(f"\n{classification_report(y_test, y_pred, target_names=['Ham','Spam'], zero_division=0)}")

# Cross-validation
cv = cross_val_score(lr, X_scaled, y, cv=5, scoring='f1_weighted')
print(f"  5-Fold CV F1: {cv.mean():.4f} ± {cv.std():.4f}")

# Save results
results_df = pd.DataFrame([{
    'Model': 'Logistic Regression',
    'Accuracy': round(acc, 4),
    'Precision': round(prec, 4),
    'Recall': round(rec, 4),
    'F1': round(f1, 4),
    'AUC_ROC': round(auc, 4),
    'CV_F1_mean': round(cv.mean(), 4),
    'CV_F1_std': round(cv.std(), 4)
}])
results_df.to_csv("../outputs/regression_results.csv", index=False)
print("\n✓ Saved: ../outputs/regression_results.csv")

# =============================================================================
# VISUALISATION 1 — Top coefficients (most spam-indicating words)
# Logistic Regression is interpretable: positive coefficient = spam word
# =============================================================================
coefs = pd.Series(lr.coef_[0], index=feature_names)

# Strip the 'tfidf_' prefix for cleaner labels
coefs.index = [n.replace('tfidf_', '') for n in coefs.index]

top_spam = coefs.nlargest(15)     # Words that most indicate SPAM
top_ham  = coefs.nsmallest(15)    # Words that most indicate HAM (legitimate)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Logistic Regression — Feature Coefficients\n'
             '(positive = spam indicator, negative = ham indicator)',
             fontsize=12, fontweight='bold')

top_spam.plot(kind='barh', ax=axes[0], color='#e74c3c')
axes[0].set_title('Top 15 SPAM Indicators', fontweight='bold')
axes[0].set_xlabel('Coefficient Value')
axes[0].axvline(0, color='black', linewidth=0.8)

top_ham.plot(kind='barh', ax=axes[1], color='#2ecc71')
axes[1].set_title('Top 15 HAM Indicators', fontweight='bold')
axes[1].set_xlabel('Coefficient Value')
axes[1].axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('../outputs/visualizations/lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Coefficient chart saved: lr_coefficients.png")

# =============================================================================
# VISUALISATION 2 — Spam probability distribution
# Shows how confident the model is — well-separated peaks = good model
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

ham_probs  = y_prob[y_test == 0]
spam_probs = y_prob[y_test == 1]

ax.hist(ham_probs,  bins=40, alpha=0.6, color='#2ecc71', label='Ham (actual)',  density=True)
ax.hist(spam_probs, bins=40, alpha=0.6, color='#e74c3c', label='Spam (actual)', density=True)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision threshold (0.5)')
ax.set_xlabel('Predicted Spam Probability')
ax.set_ylabel('Density')
ax.set_title('Logistic Regression — Spam Probability Distribution',
             fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('../outputs/visualizations/lr_probability_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("✓ Probability distribution chart saved: lr_probability_distribution.png")

# =============================================================================
# VISUALISATION 3 — ROC Curve
# AUC-ROC shows model performance across all decision thresholds
# =============================================================================
fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color='#e74c3c', linewidth=2,
        label=f'Logistic Regression (AUC = {auc:.4f})')
ax.plot([0,1], [0,1], color='gray', linestyle='--', label='Random baseline')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Logistic Regression', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/visualizations/lr_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ ROC curve saved: lr_roc_curve.png")

# =============================================================================
# VISUALISATION 4 — Confusion matrix
# =============================================================================
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'], ax=ax)
ax.set_title('Logistic Regression — Confusion Matrix', fontweight='bold')
ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('../outputs/visualizations/lr_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix saved: lr_confusion_matrix.png")

# =============================================================================
# Save model
# =============================================================================
with open("../outputs/models/logistic_regression_spam.pkl", "wb") as f:
    pickle.dump({'model': lr, 'scaler': scaler, 'feature_names': feature_names}, f)
print("✓ Model saved: logistic_regression_spam.pkl")

# =============================================================================
# Demo — show example predictions with probability scores
# =============================================================================
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS (from test set)")
print("="*60)
print(f"{'Actual':<10} {'Predicted':<12} {'Spam Prob':>10}  {'Correct?'}")
print("─"*50)

label_map = {0: 'Ham', 1: 'Spam'}
for i in range(min(10, len(y_test))):
    actual    = label_map[y_test[i]]
    predicted = label_map[y_pred[i]]
    prob      = y_prob[i]
    correct   = "✓" if y_test[i] == y_pred[i] else "✗"
    print(f"{actual:<10} {predicted:<12} {prob:>10.4f}  {correct}")

print("\n✅ Logistic Regression done!")
print("   Next: python 08_run_all_models.py  (runs everything in order)")
