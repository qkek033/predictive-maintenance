"""
FR-07: 위험도 판단.
RUL 구간별로 SAFE / WARNING / CRITICAL 분류. threshold는 설정값으로 관리하며,
분류 결과에 대한 운영 기준 해석을 제공한다.
"""
from typing import Union

import pandas as pd

# 상태 라벨 (요구사항)
RISK_SAFE = "SAFE"
RISK_WARNING = "WARNING"
RISK_CRITICAL = "CRITICAL"

# 기본 threshold (설정 가능하도록 상수로 두고, 함수 인자로 override)
DEFAULT_WARNING_THRESHOLD = 30
DEFAULT_CRITICAL_THRESHOLD = 10

# 운영 기준 정의 (요구사항)
INTERPRETATION = {
    RISK_SAFE: "정상 운영",
    RISK_WARNING: "점검 필요 (예방 정비 권장)",
    RISK_CRITICAL: "즉시 유지보수 필요",
}


def classify_risk(
    rul: Union[float, int],
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> str:
    """
    RUL 값 하나에 대해 위험도 상태를 반환한다.
    threshold는 설정값으로 인자에서 지정한다.

    기준: RUL > warning_threshold -> SAFE,
          critical_threshold < RUL <= warning_threshold -> WARNING,
          RUL <= critical_threshold -> CRITICAL
    """
    if critical_threshold >= warning_threshold:
        raise ValueError("critical_threshold는 warning_threshold보다 작아야 합니다.")
    if rul > warning_threshold:
        return RISK_SAFE
    if rul > critical_threshold:
        return RISK_WARNING
    return RISK_CRITICAL


def get_risk_interpretation(status: str) -> str:
    """상태 라벨에 대한 운영 기준 설명을 반환한다 (실무 해석용)."""
    return INTERPRETATION.get(status, "알 수 없음")


def add_risk_to_predictions(
    df: pd.DataFrame,
    rul_col: str = "predicted_rul",
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
) -> pd.DataFrame:
    """
    예측 결과 DataFrame에 위험도 분류 및 운영 기준 해석 컬럼을 붙인다.
    기존 df를 복사해 반환하며, risk_status / risk_interpretation 컬럼을 추가한다.
    """
    out = df.copy()
    out["risk_status"] = out[rul_col].map(
        lambda rul: classify_risk(rul, warning_threshold, critical_threshold)
    )
    out["risk_interpretation"] = out["risk_status"].map(get_risk_interpretation)
    return out
