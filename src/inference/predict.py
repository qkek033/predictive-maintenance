"""
FR-06: 테스트 추론.
각 unit_id의 마지막 cycle만 사용하여 RUL 예측. 학습 시 만든 feature 스키마를 그대로 사용하며,
테스트 데이터로 스키마를 재계산하지 않는다. RUL_FD001.txt와 비교 가능, 예측 결과 CSV 저장.
"""
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.engineer import get_features
from src.inference.risk import add_risk_to_predictions

logger = logging.getLogger(__name__)


def _last_cycle_per_unit(
    df: pd.DataFrame,
    unit_id_col: str = "unit_id",
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """각 unit_id별 마지막(최대) cycle 행만 남긴다. 모든 행이 아닌 unit당 1행."""
    idx = df.groupby(unit_id_col)[cycle_col].idxmax()
    return df.loc[idx].sort_values(unit_id_col).reset_index(drop=True)


def load_rul_ground_truth(rul_path: str | Path) -> pd.DataFrame:
    """
    RUL_FD001.txt 형식 파일을 로드한다. 한 줄에 한 unit의 실제 RUL.
    순서는 unit_id 1, 2, 3, ... 순으로 가정한다.

    Returns
    -------
    pd.DataFrame
        columns: unit_id, true_rul
    """
    path = Path(rul_path)
    if not path.exists():
        raise FileNotFoundError(f"RUL ground truth 파일 없음: {path}")
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    values = [int(line.strip()) for line in raw if line.strip()]
    return pd.DataFrame({"unit_id": np.arange(1, len(values) + 1, dtype=int), "true_rul": values})


def run_test_inference(
    model: Any,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    output_csv_path: str | Path,
    unit_id_col: str = "unit_id",
    cycle_col: str = "cycle",
    ground_truth_path: str | Path | None = None,
    risk_warning_threshold: float | None = None,
    risk_critical_threshold: float | None = None,
) -> pd.DataFrame:
    """
    테스트 데이터에서 unit_id별 마지막 cycle만 사용해 RUL 예측 후 CSV로 저장한다.
    feature 스키마는 학습에서 만든 feature_columns를 그대로 사용하며, 테스트로 재계산하지 않는다.

    Parameters
    ----------
    model : fitted regressor
        predict() 메서드를 가진 학습된 모델
    test_df : pd.DataFrame
        test_FD001 로드 결과 (전체 행). 내부에서 마지막 cycle만 필터한다.
    feature_columns : List[str]
        학습 시 prepare_feature_schema()로 얻은 컬럼 목록 (동일 스키마 유지)
    output_csv_path : str | Path
        예측 결과를 저장할 CSV 경로
    unit_id_col : str
        유닛 식별자 컬럼명
    cycle_col : str
        사이클 컬럼명
    ground_truth_path : str | Path | None
        지정 시 RUL_FD001.txt 경로로 읽어 true_rul과 비교 후 RMSE/MAE 로그 및 CSV에 포함
    risk_warning_threshold : float | None
        위험도 WARNING 구간 상한 (RUL > 이 값이면 SAFE). None 이면 위험도 컬럼 미추가
    risk_critical_threshold : float | None
        위험도 CRITICAL 구간 상한 (RUL <= 이 값이면 CRITICAL). None 이면 위험도 컬럼 미추가

    Returns
    -------
    pd.DataFrame
        unit_id, predicted_rul [, true_rul] [, risk_status, risk_interpretation] 컬럼을 가진 결과
    """
    missing = [c for c in feature_columns if c not in test_df.columns]
    if missing:
        raise ValueError(f"테스트 데이터에 feature 컬럼 없음 (학습 스키마와 불일치): {missing}")

    last_df = _last_cycle_per_unit(test_df, unit_id_col=unit_id_col, cycle_col=cycle_col)
    logger.info("테스트 추론: unit_id별 마지막 cycle만 사용, 행 수=%s (전체 %s 아님)", len(last_df), len(test_df))

    X = get_features(last_df, feature_columns)
    pred = model.predict(X)
    result = pd.DataFrame({unit_id_col: last_df[unit_id_col].values, "predicted_rul": pred})

    if ground_truth_path is not None:
        gt = load_rul_ground_truth(ground_truth_path)
        result = result.merge(gt, on=unit_id_col, how="left")
        if result["true_rul"].notna().all():
            rmse = np.sqrt(mean_squared_error(result["true_rul"], result["predicted_rul"]))
            mae = mean_absolute_error(result["true_rul"], result["predicted_rul"])
            logger.info("RUL_FD001 비교: RMSE=%.4f MAE=%.4f", rmse, mae)
        else:
            logger.warning("일부 unit에 true_rul 없음, 비교 생략")

    if risk_warning_threshold is not None and risk_critical_threshold is not None:
        result = add_risk_to_predictions(
            result,
            rul_col="predicted_rul",
            warning_threshold=risk_warning_threshold,
            critical_threshold=risk_critical_threshold,
        )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv_path, index=False, encoding="utf-8")
    logger.info("예측 결과 저장: %s", output_csv_path)
    return result
