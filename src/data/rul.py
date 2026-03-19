"""
FR-02: RUL(Remaining Useful Life) 생성.
각 unit_id별 최대 cycle 기준으로 RUL = max_cycle - cycle 를 계산하며,
설정 가능한 상한 cap 및 분포/이상치 점검을 지원한다.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# RUL 컬럼명
RUL_COLUMN = "RUL"


def add_rul(
    df: pd.DataFrame,
    cycle_col: str = "cycle",
    unit_id_col: str = "unit_id",
    cap: int | None = None,
) -> pd.DataFrame:
    """
    unit_id별 최대 cycle을 구한 뒤 RUL = max_cycle - cycle 로 계산하여 컬럼을 추가한다.
    학습용 run-to-failure 데이터프레임에만 사용하는 것을 전제로 한다.

    Parameters
    ----------
    df : pd.DataFrame
        unit_id, cycle 컬럼을 가진 데이터프레임 (예: train_FD001 로드 결과)
    cycle_col : str
        사이클 번호 컬럼명
    unit_id_col : str
        유닛 식별자 컬럼명
    cap : int | None
        RUL 상한. 지정 시 RUL = min(RUL, cap) 로 제한한다. None 이면 cap 미적용.

    Returns
    -------
    pd.DataFrame
        RUL 컬럼이 추가된 복사본
    """
    if unit_id_col not in df.columns or cycle_col not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {unit_id_col}, {cycle_col}")

    out = df.copy()

    # unit_id별 최대 cycle 계산 (각 행에 해당 unit의 max_cycle 부여)
    max_cycle = out.groupby(unit_id_col)[cycle_col].transform("max")
    rul = max_cycle - out[cycle_col]

    # 상한 cap 적용 (configurable)
    if cap is not None:
        rul = rul.clip(upper=cap)
        logger.info("RUL 상한 cap 적용: cap=%s", cap)

    out[RUL_COLUMN] = rul.astype(int)

    # RUL 분포 확인 및 이상 값 점검
    _log_rul_distribution(out[RUL_COLUMN])
    _check_rul_anomalies(out[RUL_COLUMN], unit_id_col, cycle_col, out)

    logger.info("RUL 생성 완료: shape=%s, RUL min=%s max=%s", out.shape, out[RUL_COLUMN].min(), out[RUL_COLUMN].max())
    return out


def _log_rul_distribution(rul: pd.Series) -> None:
    """RUL 분포를 로그로 출력한다."""
    stats = rul.describe()
    logger.info("RUL 분포: count=%s mean=%.2f std=%.2f min=%s 25%%=%.0f 50%%=%.0f 75%%=%.0f max=%s",
                stats["count"], stats["mean"], stats["std"],
                stats["min"], stats["25%"], stats["50%"], stats["75%"], stats["max"])


def _check_rul_anomalies(
    rul: pd.Series, unit_id_col: str, cycle_col: str, df: pd.DataFrame
) -> None:
    """RUL 이상 값 점검: 음수 여부, unit_id별 마지막 cycle에서 RUL=0 여부를 로그한다."""
    n_negative = (rul < 0).sum()
    if n_negative > 0:
        logger.warning("RUL 이상치: 음수 RUL 개수=%s (정상이라면 0이어야 함)", n_negative)

    # unit_id별 마지막 cycle에서 RUL이 0인지 샘플 확인 (선택적 점검)
    last_cycles = df.groupby(unit_id_col)[cycle_col].max()
    for uid, last_c in last_cycles.items():
        row = df[(df[unit_id_col] == uid) & (df[cycle_col] == last_c)]
        if not row.empty and row[RUL_COLUMN].iloc[0] != 0:
            logger.warning("unit_id=%s 마지막 cycle(%s)에서 RUL=%s (기대값 0)", uid, last_c, row[RUL_COLUMN].iloc[0])
            break  # 한 건만 로그
