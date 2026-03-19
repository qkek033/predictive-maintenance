"""
학습 실행 후 모델·스키마를 output/model에 저장.
API 서버는 이 경로에서 아티팩트를 로드한다. 기존 ML 파이프라인만 사용.
"""
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.artifact import get_artifact_dir, save_artifact
from src.data import add_rul, load_train_fd001
from src.features import prepare_feature_schema
from src.models import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    artifact_dir = get_artifact_dir()
    logger.info("학습 후 아티팩트 저장 경로: %s", artifact_dir)

    train = load_train_fd001()
    train = add_rul(train, cap=125)
    schema = prepare_feature_schema(train)
    model = train_model(train, schema, seed=42)

    save_artifact(model, schema, artifact_dir)
    logger.info("저장 완료: model.joblib, feature_columns.json")


if __name__ == "__main__":
    main()
