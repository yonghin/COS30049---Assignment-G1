from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Request, HTTPException
from backend.history_store import history_store

router = APIRouter()

# Values sourced from outputs/classification_results.csv and
# outputs/validation/model_ranking.csv (weighted-average precision/recall).
MODELS_METRICS = [
    {"name": "rf_spam",    "task": "Spam Detection",    "accuracy": 0.9839, "precision": 0.9839, "recall": 0.9839, "f1": 0.9839, "auc": 0.9978},
    {"name": "nb_spam",    "task": "Spam Detection",    "accuracy": 0.9671, "precision": 0.9663, "recall": 0.9671, "f1": 0.9662, "auc": 0.9787},
    {"name": "lr_spam",    "task": "Spam Detection",    "accuracy": 0.9613, "precision": 0.9610, "recall": 0.9613, "f1": 0.9591, "auc": 0.9899},
    {"name": "svm_malware","task": "Malware Detection", "accuracy": 0.9992, "precision": 0.9993, "recall": 0.9992, "f1": 0.9993, "auc": 1.0000},
]


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": ["rf_spam", "nb_spam", "logistic_regression_spam",
                          "svm_malware", "kmeans_malware", "dbscan_malware"],
    }


@router.get("/models")
async def get_models():
    return {"models": MODELS_METRICS}


@router.get("/predictions/history")
async def get_history(since: str = None):
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, "Invalid since timestamp")
    else:
        since_dt = datetime.now(timezone.utc) - timedelta(minutes=60)
    spam_series, malware_series = history_store.to_time_series(since_dt)
    return {"spam_series": spam_series, "malware_series": malware_series}


@router.delete("/predictions/history")
async def clear_history():
    history_store.clear()
    return {"message": "Prediction history cleared."}
