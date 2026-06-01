import io
import pandas as pd
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, field_validator
from backend.services.spam_service import predict_single, predict_batch
from backend.history_store import history_store, PredictionEntry
from datetime import datetime, timezone

router = APIRouter()

VALID_MODELS = {"rf_spam", "nb_spam", "logistic_regression_spam"}


class SpamPredictRequest(BaseModel):
    text: str = Field(..., min_length=3)
    model: str = Field(default="rf_spam")

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v):
        if v not in VALID_MODELS:
            raise ValueError(f"model must be one of {VALID_MODELS}")
        return v


class SpamPredictResponse(BaseModel):
    label: str
    spam_prob: float
    ham_prob: float
    confidence: float
    model_used: str
    timestamp: str


class SpamRowResult(BaseModel):
    row: int
    text: str
    label: str
    spam_prob: float


class SpamBatchResponse(BaseModel):
    total: int
    spam_count: int
    ham_count: int
    model_used: str
    results: list[SpamRowResult]


@router.post("/predict", response_model=SpamPredictResponse)
async def predict_single_endpoint(req: SpamPredictRequest, request: Request):
    result = predict_single(req.text, req.model, request.app.state.registry)
    history_store.append(PredictionEntry(
        timestamp=datetime.now(timezone.utc),
        model=req.model, task="spam",
        label=result.label, confidence=result.confidence,
    ))
    return SpamPredictResponse(**result.__dict__)


@router.post("/predict/batch", response_model=SpamBatchResponse)
async def predict_batch_endpoint(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="rf_spam"),
):
    if model not in VALID_MODELS:
        raise HTTPException(400, f"Invalid model. Choose from: {VALID_MODELS}")
    content = await file.read()
    filename = file.filename or ""
    if filename.endswith(".txt"):
        messages = [l.strip() for l in content.decode().splitlines() if l.strip()]
    elif filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        if "message" not in df.columns:
            raise HTTPException(422, "CSV must have 'message' column")
        messages = df["message"].astype(str).tolist()
    else:
        raise HTTPException(422, "File must be .txt or .csv")
    if not messages:
        raise HTTPException(422, "File contains no messages")
    results = predict_batch(messages, model, request.app.state.registry)
    for r in results:
        history_store.append(PredictionEntry(
            timestamp=datetime.now(timezone.utc),
            model=model, task="spam",
            label=r.label, confidence=r.confidence,
        ))
    spam_count = sum(1 for r in results if r.label == "SPAM")
    return SpamBatchResponse(
        total=len(results), spam_count=spam_count, ham_count=len(results) - spam_count,
        model_used=model,
        results=[SpamRowResult(row=i+1, text=m, label=r.label, spam_prob=r.spam_prob)
                 for i, (m, r) in enumerate(zip(messages, results))],
    )
