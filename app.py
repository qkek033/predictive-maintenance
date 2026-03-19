# Render 등 배포 시 "uvicorn app:app --host 0.0.0.0 --port $PORT" 로 실행.
# 배포 환경에서 프로젝트 루트가 PYTHONPATH에 없으면 "src" 패키지를 찾지 못해 라우트가 로드되지 않으므로,
# 이 파일 기준으로 루트 디렉터리를 sys.path에 넣어 import가 성공하도록 한다.
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.api.main import app


@app.get("/")
def root():
    """루트 URL. 배포 환경에서 서비스 존재 확인용. GET /health와 동일한 의미."""
    return {"status": "ok", "message": "Predictive Maintenance API"}


__all__ = ["app"]
