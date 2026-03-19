"""
FR-03: Feature Engineering.
입력 feature는 cycle, operational settings, sensor values 로 구성하고,
상수 컬럼을 제거하며 train/test 동일 스키마를 유지한다.
"""
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# 요구사항 입력 feature 구성: cycle, operational settings, sensor values (unit_id 제외)
FEATURE_COLUMNS_BASE: List[str] = (
    ["cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def prepare_feature_schema(
    train_df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> List[str]:
    """
    학습 데이터에서 상수 컬럼을 제거한 feature 컬럼 목록을 반환한다.
    반환된 목록은 train/test 동일 스키마로 get_features()에 넘겨 재사용한다.

    Parameters
    ----------
    train_df : pd.DataFrame
        학습용 데이터프레임 (cycle, op_setting_*, sensor_* 컬럼 포함)
    feature_cols : List[str] | None
        후보 feature 컬럼. None 이면 FEATURE_COLUMNS_BASE 사용.

    Returns
    -------
    List[str]
        상수 컬럼 제거 후 남은 feature 컬럼 목록
    """
    cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS_BASE.copy()
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"학습 데이터에 feature 컬럼 없음: {missing}")

    # 상수 컬럼 제거 (분산 0 또는 고유값 1개)
    non_constant = []
    dropped = []
    for c in cols:
        if train_df[c].nunique() <= 1:
            dropped.append(c)
        else:
            non_constant.append(c)

    if dropped:
        logger.info("상수 컬럼 제거: %s", dropped)
    logger.info("feature 스키마: %s개 컬럼", len(non_constant))
    return non_constant


def get_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    지정한 feature 컬럼만 추출한다. train/test 동일 스키마를 위해 prepare_feature_schema()에서
    얻은 목록을 그대로 사용해야 한다.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임
    feature_columns : List[str]
        prepare_feature_schema()로 얻은 feature 컬럼 목록

    Returns
    -------
    pd.DataFrame
        feature만 포함한 복사본 (컬럼 순서 동일)
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"데이터에 feature 컬럼 없음: {missing}")
    out = df[feature_columns].copy()
    logger.debug("get_features: shape=%s", out.shape)
    return out
