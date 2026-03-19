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
    """
    입력: {"features": {"cycle": 100, "op_setting_1": 0.5, ...}} (feature 이름을 키로 하는 dict).
    출력: {"prediction": RUL 예측값, "risk": "SAFE"|"WARNING"|"CRITICAL"}.
    필수 컬럼 누락·허용되지 않은 키·타입 오류 시 422. 기존 모델·inference.risk만 재사용.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. 학습 후 아티팩트를 저장하세요.")
    raw = body.get("features")
    if raw is None:
        raise HTTPException(status_code=422, detail="필수 키 누락: features")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="features는 객체(dict)여야 합니다.")
    # 필수 컬럼 누락 검사
    missing = [c for c in _feature_columns if c not in raw]
    if missing:
        raise HTTPException(status_code=422, detail=f"필수 feature 누락: {missing}")
    # 허용되지 않은 키(extra) 검사
    extra = [k for k in raw if k not in _feature_columns]
    if extra:
        raise HTTPException(status_code=422, detail=f"허용되지 않은 feature(제거 필요): {extra}")
    # 값 타입 검사: 모두 숫자여야 함
    for col in _feature_columns:
        v = raw[col]
        if v is None or not isinstance(v, (int, float)):
            raise HTTPException(status_code=422, detail=f"feature 타입 오류(숫자 아님): {col}")
    # 스키마 순서대로 1행 DataFrame 생성 후 기존 모델로 예측 (컬럼 정렬로 재현성 유지)
    row = {c: raw[c] for c in _feature_columns}
    X = pd.DataFrame([row])
    prediction = float(_model.predict(X)[0])
    risk = classify_risk(prediction)
    return {"prediction": prediction, "risk": risk}
