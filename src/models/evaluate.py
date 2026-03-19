"""
FR-05: 모델 평가.
기존 학습된 모델로 RMSE, MAE 계산 및 feature importance 출력.
결과를 로그 및 파일로 저장한다.
"""
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    feature_columns: List[str] | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """
    이미 학습된 모델로 예측 후 RMSE, MAE를 계산하고 feature importance를 출력한다.
    모델을 재학습하지 않는다.

    Parameters
    ----------
    model : fitted regressor
        predict() 및 (선택) feature_importances_ 속성을 가진 모델
    X : pd.DataFrame
        평가할 feature 데이터 (컬럼 순서는 feature_columns와 일치해야 함)
    y_true : pd.Series
        실제 RUL 값
    feature_columns : List[str] | None
        X의 컬럼 순서. None 이면 X.columns 사용 (feature importance용)
    output_path : str | Path | None
        지정 시 평가 결과를 이 경로에 저장

    Returns
    -------
    dict
        {"rmse": float, "mae": float, "feature_importance": list of (name, value)}
    """
    cols = feature_columns if feature_columns is not None else list(X.columns)
    pred = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))

    logger.info("평가 RMSE=%.4f MAE=%.4f", rmse, mae)

    importance_list: List[tuple] = []
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        if len(imp) == len(cols):
            importance_list = list(zip(cols, imp.tolist()))
            importance_list.sort(key=lambda x: x[1], reverse=True)
            for name, val in importance_list[:10]:
                logger.info("  importance %s: %.2f", name, val)
            if len(importance_list) > 10:
                logger.info("  ... (%s개 feature)", len(importance_list))
        else:
            logger.warning("feature_importances_ 길이와 컬럼 수 불일치, importance 생략")

    results = {"rmse": rmse, "mae": mae, "feature_importance": importance_list}

    if output_path is not None:
        _save_evaluation_results(results, Path(output_path))

    return results


def _save_evaluation_results(results: dict, path: Path) -> None:
    """평가 결과를 읽기 쉬운 텍스트 형식으로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "========================================",
        "  Model evaluation results",
        "========================================",
        "",
        "Metrics",
        "-------",
        "  RMSE: {:.4f}".format(results["rmse"]),
        "  MAE:  {:.4f}".format(results["mae"]),
        "",
    ]
    if results.get("feature_importance"):
        lines.append("Feature importance (descending)")
        lines.append("-----------------------------")
        for name, val in results["feature_importance"]:
            lines.append("  {:25s}  {:>10.4f}".format(name, val))
        lines.append("")
    lines.append("========================================")

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    logger.info("평가 결과 저장: %s", path)
