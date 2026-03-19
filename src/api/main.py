"""
FR-08: 추론 API. GET /health, POST /predict.
기존 아티팩트(모델·스키마)와 inference.risk만 재사용한다.
"""
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.artifact import load_artifact
from src.inference.risk import classify_risk

_model: Any = None
_feature_columns: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 저장된 모델·스키마 로드. 없으면 /predict에서 503."""
    global _model, _feature_columns
    try:
        _model, _feature_columns = load_artifact()
    except FileNotFoundError:
        pass
    yield
    _model, _feature_columns = None, []


app = FastAPI(title="Predictive Maintenance API", lifespan=lifespan)


@app.get("/health")
def health():
    """서비스 상태 반환."""
    return {"status": "ok"}


@app.post("/predict")
def predict(body: dict):
    """feature 값을 받아 predicted_rul, risk_status 반환. 필수 feature 누락·타입 오류 시 422."""
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. 학습 후 아티팩트를 저장하세요.")
    missing = [c for c in _feature_columns if c not in body]
    if missing:
        raise HTTPException(status_code=422, detail=f"필수 feature 누락: {missing}")
    for col in _feature_columns:
        v = body[col]
        if v is None or not isinstance(v, (int, float)):
            raise HTTPException(status_code=422, detail=f"feature 타입 오류(숫자 아님): {col}")
    row = {c: body[c] for c in _feature_columns}
    X = pd.DataFrame([row])
    predicted_rul = float(_model.predict(X)[0])
    risk_status = classify_risk(predicted_rul)
    return {"predicted_rul": predicted_rul, "risk_status": risk_status}
