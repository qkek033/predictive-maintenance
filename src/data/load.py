"""
FR-01: NASA CMAPSS FD001 데이터 로딩.
data/raw/ 경로의 train_FD001.txt, test_FD001.txt를 공백 구분자로 로드한다.
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 요구사항 컬럼 구성: unit_id, cycle, op_setting_1~3, sensor_1~sensor_21
RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _load_fd001_txt(path: Path) -> pd.DataFrame:
    """FD001 형식 txt 파일을 공백 구분자로 읽고, 끝쪽 빈 컬럼을 제거한다."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
        engine="python",
    )
    # 불필요한 trailing empty column 제거
    df = df.loc[:, df.notna().any(axis=0)]
    return df


def load_train_fd001(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """학습용 데이터를 data_dir/train_FD001.txt에서 로드한다."""
    data_dir = Path(data_dir)
    path = data_dir / "train_FD001.txt"
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    df = _load_fd001_txt(path)
    logger.info("train_FD001 loaded: shape=%s", df.shape)
    logger.debug("train_FD001 head:\n%s", df.head())
    return df


def load_test_fd001(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """테스트 데이터를 data_dir/test_FD001.txt에서 로드한다."""
    data_dir = Path(data_dir)
    path = data_dir / "test_FD001.txt"
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    df = _load_fd001_txt(path)
    logger.info("test_FD001 loaded: shape=%s", df.shape)
    logger.debug("test_FD001 head:\n%s", df.head())
    return df
