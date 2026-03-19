"""
FR-04: 모델 학습.
RUL 회귀용 모델 학습, unit_id 기준 분할(data leakage 방지), random seed 고정.
기본 LightGBM, fallback RandomForest.
"""
import logging
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# 재현성을 위한 기본 시드
DEFAULT_SEED = 42


def _split_unit_ids(
    unit_ids: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """unit_id 목록을 train_ratio 비율로 나눈다. 행 단위가 아닌 unit 단위 분할."""
    uniq = np.unique(unit_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = int(len(uniq) * train_ratio)
    return uniq[:n], uniq[n:]


def train_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_col: str = "RUL",
    unit_id_col: str = "unit_id",
    train_ratio: float = 0.8,
    seed: int = DEFAULT_SEED,
    model_type: str = "lgb",
) -> Any:
    """
    unit_id 기준으로 train/val을 나눈 뒤 RUL 회귀 모델을 학습한다.
    행 단위 무작위 분할을 사용하지 않는다.

    Parameters
    ----------
    df : pd.DataFrame
        RUL 및 feature 컬럼을 포함한 학습용 데이터프레임 (unit_id 필요)
    feature_columns : List[str]
        모델 입력 feature 컬럼 목록 (prepare_feature_schema 결과)
    target_col : str
        목표 컬럼명 (기본 RUL)
    unit_id_col : str
        유닛 식별자 컬럼명
    train_ratio : float
        학습에 쓸 unit 비율 (나머지는 검증)
    seed : int
        unit 분할 및 모델 내부 난수 시드
    model_type : str
        "lgb" (기본) 또는 "rf" (fallback)

    Returns
    -------
    fitted model (LightGBM 또는 RandomForest Regressor)
    """
    if unit_id_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {unit_id_col}, {target_col}")
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"feature 컬럼 없음: {missing}")

    train_units, val_units = _split_unit_ids(df[unit_id_col].values, train_ratio=train_ratio, seed=seed)
    train_mask = df[unit_id_col].isin(train_units)
    val_mask = df[unit_id_col].isin(val_units)

    X_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, target_col]
    X_val = df.loc[val_mask, feature_columns]
    y_val = df.loc[val_mask, target_col]

    logger.info(
        "unit_id 분할: train_units=%s val_units=%s -> train shape=%s val shape=%s",
        len(train_units), len(val_units), X_train.shape, X_val.shape,
    )

    if model_type == "lgb":
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM 없음, RandomForest 사용")
            model_type = "rf"

    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=seed, n_estimators=100)
        model.fit(X_train, y_train)
        logger.info("RandomForest 학습 완료")
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        logger.info("val RMSE=%.4f MAE=%.4f", val_rmse, val_mae)
        return model

    # LightGBM
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        random_state=seed,
        n_estimators=100,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    logger.info("LightGBM 학습 완료")
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    logger.info("val RMSE=%.4f MAE=%.4f", val_rmse, val_mae)
    return model
