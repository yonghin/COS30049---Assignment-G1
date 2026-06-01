import re
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np


@dataclass
class SpamPredictionResult:
    label: str
    spam_prob: float
    ham_prob: float
    confidence: float
    model_used: str
    timestamp: str


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _predict_rf(text: str, registry: dict) -> tuple[float, float]:
    rf_pkg = registry["rf_spam"]
    rf = rf_pkg["model"]
    feature_cols = rf_pkg["feature_cols"]
    cleaned = clean_text(text)
    feature_values = []
    for col in feature_cols:
        if col == "message_length":
            feature_values.append(len(text))
        elif col == "word_count":
            feature_values.append(len(text.split()))
        elif col.startswith("has_"):
            kw = col[4:]
            feature_values.append(1 if kw in cleaned.split() else 0)
        else:
            feature_values.append(0)
    X = np.array(feature_values).reshape(1, -1)
    probs = rf.predict_proba(X)[0]
    return float(probs[0]), float(probs[1])  # ham_p, spam_p


def _predict_tfidf_model(text: str, registry: dict, model_key: str) -> tuple[float, float]:
    cleaned = clean_text(text)
    tfidf = registry["spam_tfidf"]
    pkg = registry[model_key]
    model = pkg["model"]
    scaler = pkg["scaler"]
    X_tfidf = tfidf.transform([cleaned]).toarray()
    X_scaled = scaler.transform(X_tfidf)
    probs = model.predict_proba(X_scaled)[0]
    return float(probs[0]), float(probs[1])  # ham_p, spam_p


def predict_single(text: str, model_name: str, registry: dict) -> SpamPredictionResult:
    if model_name == "rf_spam":
        ham_p, spam_p = _predict_rf(text, registry)
    elif model_name == "nb_spam":
        ham_p, spam_p = _predict_tfidf_model(text, registry, "nb_spam")
    elif model_name == "logistic_regression_spam":
        ham_p, spam_p = _predict_tfidf_model(text, registry, "lr_spam")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    label = "SPAM" if spam_p >= 0.5 else "HAM"
    confidence = max(spam_p, ham_p)
    return SpamPredictionResult(
        label=label, spam_prob=spam_p, ham_prob=ham_p,
        confidence=confidence, model_used=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def predict_batch(messages: list, model_name: str, registry: dict) -> list:
    if not messages:
        return []
    return [predict_single(m, model_name, registry) for m in messages]
