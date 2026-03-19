# Render 등 배포 시 "uvicorn app:app --host 0.0.0.0 --port 10000" (또는 PORT 환경변수)로
# 실행할 수 있도록 루트에서 FastAPI app을 노출한다. 실제 앱은 src.api.main에 정의되어 있음.
from src.api.main import app

__all__ = ["app"]
