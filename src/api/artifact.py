"""
추론 API용 아티팩트(모델, feature 스키마) 저장/로드.
기존 ML 파이프라인과 분리하여 배포 시 한 디렉터리에서 로드한다.
"""
import json
import os
from pathlib import Path
from typing import Any, List

import joblib


def get_artifact_dir() -> Path:
    """아티팩트 경로: 환경변수 MODEL_DIR 또는 기본 output/model."""
    return Path(os.environ.get("MODEL_DIR", "output/model"))


def save_artifact(
    model: Any,
    feature_columns: List[str],
    artifact_dir: str | Path | None = None,
) -> None:
    """모델(model.joblib)과 feature 스키마(feature_columns.json) 저장. 학습 직후 한 번 호출."""
    d = Path(artifact_dir) if artifact_dir else get_artifact_dir()
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib")
    (d / "feature_columns.json").write_text(json.dumps(feature_columns, ensure_ascii=False), encoding="utf-8")


def load_artifact(artifact_dir: str | Path | None = None) -> tuple[Any, List[str]]:
    """저장된 모델·스키마 로드. 파일 없으면 FileNotFoundError."""
    d = Path(artifact_dir) if artifact_dir else get_artifact_dir()
    model_path = d / "model.joblib"
    schema_path = d / "feature_columns.json"
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}. 학습 후 save_artifact를 실행하세요.")
    if not schema_path.exists():
        raise FileNotFoundError(f"스키마 파일 없음: {schema_path}")
    model = joblib.load(model_path)
    feature_columns = json.loads(schema_path.read_text(encoding="utf-8"))
    return model, feature_columns
