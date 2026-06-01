from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()

ALLOWED_MODELS = {"rf_spam", "nb_spam", "logistic_regression_spam", "svm_malware"}


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class RocData(BaseModel):
    fpr: list[float]
    tpr: list[float]
    auc: float


class ModelAnalyticsResponse(BaseModel):
    model: str
    confusion_matrix: list[list[int]]
    roc: RocData
    feature_importance: list[FeatureImportanceItem] | None


@router.get("/model/{model_name}", response_model=ModelAnalyticsResponse)
async def get_model_analytics(model_name: str, request: Request):
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(404, f"Unknown model: {model_name}")
    data = request.app.state.analytics.get(model_name)
    if data is None:
        raise HTTPException(503, "Analytics not available for this model")
    return ModelAnalyticsResponse(**data)
