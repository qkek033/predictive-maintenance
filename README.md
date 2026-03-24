# 예지보전 · RUL 예측 (NASA CMAPSS FD001)

**LightGBM** 회귀 → **FastAPI** 추론 → **Streamlit** 대시보드.

---

## 프로젝트 개요

- **문제**: 센서·운전 데이터만으로 잔여 수명(RUL)과 정비 우선순위를 바로 쓰기 어렵다.
- **해결**: C-MAPSS로 RUL 회귀 모델을 학습하고, 단일 시점 feature를 **API**로 받아 RUL·**SAFE / WARNING / CRITICAL** 를 반환한 뒤 **대시보드**에서 시각화한다.
- **가치**: RUL·위험도를 선제적으로 보면 **다운타임과 긴급 정비 비용**을 줄이고, 계획 정비로 운영 리스크를 낮출 수 있다.

---

## 주요 기능

- **RUL prediction** — 사이클 단위 잔여 수명 회귀
- **Risk classification** — `classify_risk` 기반 `SAFE` · `WARNING` · `CRITICAL`
- **FastAPI** — `GET /health`, `POST /predict`, JSON 스키마 검증
- **Streamlit dashboard** — feature 입력 후 배포 API 호출
- **Visualization** — Plotly gauge, RUL 추이 라인 차트(시뮬레이션)

---

## 시스템 구조

**Model (artifact) → FastAPI → Streamlit** — 한 줄로는 학습 산출물을 API가 로드하고, 대시보드가 HTTP로 `/predict`만 호출한다.

---

## 대시보드

- **Gauge chart** — RUL 구간별 색·눈금(Plotly)
- **RUL trend graph** — cycle–RUL 라인(시뮬레이션 추이, 캡션 안내)
- **Risk visualization** — API `risk`·구간 안내·메트릭으로 위험도 표시

---

## 배포

- **FastAPI** — **Render** 등에 배포 (`app.py` 또는 `uvicorn` 엔트리)
- **Streamlit** — **Render** 등에서 `streamlit_app.py` 실행, `API_URL`을 백엔드와 일치

---

## 🔗 Demo

- **Dashboard** — Streamlit 공개 URL은 **Render(또는 사용 중인 호스팅)** 의 대시보드 서비스 주소와 동일(저장소에 하드코딩 없음).
- **API** — [https://predictive-maintenance-1-hoz8.onrender.com](https://predictive-maintenance-1-hoz8.onrender.com) (`GET /health`, `POST /predict`)

---

## 설계 의도

- **Gauge** — RUL은 숫자보다 위험 구간이 의사결정에 직결; 한눈에 구간 인지
- **시각화** — API만으로도 동작하나 데모·설명을 위해 맥락을 한 화면에
- **API + UI 분리** — 동일 계약으로 다른 클라이언트 재사용, 배포·스케일 분리

---

## 📄 상세 문서

더 자세한 내용은 아래 문서를 참고하세요:

- [Full Documentation](docs/full_readme.md)
