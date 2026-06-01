import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

logger = logging.getLogger(__name__)


def initialize(registry: dict) -> dict:
    cache = {}
    cache["rf_spam"]                  = _compute_rf_spam(registry)
    cache["nb_spam"]                  = _compute_nb_spam(registry)
    cache["logistic_regression_spam"] = _compute_lr_spam(registry)
    cache["svm_malware"]              = _compute_svm_malware(registry)
    return cache


def _build_result(model_name, y_test, y_pred, y_prob, feature_importance=None):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = float(roc_auc_score(y_test, y_prob))
    return {
        "model": model_name,
        "confusion_matrix": cm,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc},
        "feature_importance": feature_importance,
    }


def _compute_rf_spam(registry):
    try:
        df = pd.read_csv("data/processed/combined_spam_processed.csv")
    except FileNotFoundError:
        logger.warning("combined_spam_processed.csv not found; rf_spam analytics unavailable")
        return None
    feature_cols = registry["rf_spam"]["feature_cols"]
    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = registry["rf_spam"]["model"]
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    importances = sorted(
        [{"feature": f, "importance": float(i)} for f, i in zip(feature_cols, rf.feature_importances_)],
        key=lambda x: x["importance"], reverse=True,
    )
    return _build_result("rf_spam", y_test, y_pred, y_prob, importances)


def _compute_nb_spam(registry):
    try:
        df = pd.read_csv("data/processed/sms_spam_tfidf.csv")
    except FileNotFoundError:
        logger.warning("sms_spam_tfidf.csv not found; nb_spam analytics unavailable")
        return None
    y = df["label_encoded"].values
    X = df.drop("label_encoded", axis=1).values
    X = registry["nb_spam"]["scaler"].transform(X)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    nb = registry["nb_spam"]["model"]
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]
    return _build_result("nb_spam", y_test, y_pred, y_prob, None)


def _compute_lr_spam(registry):
    try:
        df = pd.read_csv("data/processed/sms_spam_tfidf.csv")
    except FileNotFoundError:
        logger.warning("sms_spam_tfidf.csv not found; lr_spam analytics unavailable")
        return None
    y = df["label_encoded"].values
    X = df.drop("label_encoded", axis=1).values
    X = registry["lr_spam"]["scaler"].transform(X)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr = registry["lr_spam"]["model"]
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    feature_names = registry["lr_spam"]["feature_names"]
    coefs = np.abs(lr.coef_[0])
    top20 = sorted(
        [{"feature": f, "importance": float(c)} for f, c in zip(feature_names, coefs)],
        key=lambda x: x["importance"], reverse=True,
    )[:20]
    return _build_result("logistic_regression_spam", y_test, y_pred, y_prob, top20)


def _compute_svm_malware(registry):
    try:
        df = pd.read_csv("data/processed/malmem_processed.csv")
    except FileNotFoundError:
        logger.warning("malmem_processed.csv not found; svm_malware analytics unavailable")
        return None
    drop_cols = [c for c in ["binary_label", "category_encoded", "category_name"] if c in df.columns]
    y = df["binary_label"].values
    X = df.drop(columns=drop_cols).values
    if len(X) > 20000:
        idx = np.random.RandomState(42).choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm = registry["svm_malware"]
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]
    return _build_result("svm_malware", y_test, y_pred, y_prob, None)
