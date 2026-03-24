# 예지보전 시스템 (Predictive Maintenance) — NASA CMAPSS RUL 예측

NASA C-MAPSS FD001 데이터를 기반으로 **잔여 수명(RUL)** 을 예측하고, **위험도(SAFE / WARNING / CRITICAL)** 를 분류합니다.  
**LightGBM** 회귀로 학습한 모델을 **FastAPI**로 제공하고, **Streamlit** 대시보드에서 입력·예측·시각화까지 한 흐름으로 사용할 수 있습니다.

---

## 프로젝트 개요

### 문제 정의

- 센서·운전 데이터가 있어도 **“지금 남은 수명이 어느 정도인지”** 를 바로 쓰기 어렵습니다.
- 회귀값(RUL)만 있으면 **현장 조치(점검·정비 우선순위)** 로 연결하기 위해 추가 해석이 필요합니다.

### 해결 방법

- Run-to-Failure 학습 데이터로 **RUL 회귀 모델**을 학습하고, 시점별 feature로 RUL을 추정합니다.
- 배치·테스트와 동일하게 **유닛의 마지막 cycle(또는 단일 시점)** 에 해당하는 feature 벡터를 **API 입력**으로 받아 추론합니다.
- API는 `prediction`과 함께 **risk** 문자열을 반환하고, Streamlit에서 **게이지·추이 차트**로 상태를 시각화합니다.

### 예지보전의 가치

- 고장 **이후** 수리보다 **이전** 계획 정비로 전환해 다운타임·긴급 비용을 줄입니다.
- 데이터 기반으로 **어느 설비를 먼저 볼지** 우선순위를 정할 수 있습니다.

---

## 주요 기능

| 구분 | 설명 |
|------|------|
| **RUL prediction** | Remaining Useful Life, 사이클 단위 회귀 예측 |
| **Risk classification** | 예측 RUL에 따른 `SAFE` · `WARNING` · `CRITICAL` (API: `classify_risk`) |
| **FastAPI** | `GET /health`, `POST /predict` — JSON in/out, 스키마 검증 |
| **Streamlit dashboard** | feature 입력 → 배포 API 호출 → 결과·차트 표시 |
| **Visualization** | Plotly **gauge chart** (RUL 구간), **RUL trend** 라인 차트(시뮬레이션 추이) |

---

## 시스템 구조

**Model → API → Frontend** 한 줄 흐름입니다.

```text
[ 학습 아티팩트: 모델 + feature column 목록 ]
            ↓ FastAPI startup 시 로드
[ POST /predict ]  body: { "features": { "cycle": ..., "sensor_2": ..., ... } }
            ↓
{ "prediction": <float>, "risk": "SAFE"|"WARNING"|"CRITICAL" }
            ↓ HTTP (Streamlit urllib)
[ Streamlit ]  좌측 입력 / 우측 결과·게이지·추이
```

- **Model / 학습**: `src/models`, `src/features`, `src/data`  
- **Serving**: `src/api` (추론 시 `inference.risk` 재사용)  
- **UI**: `streamlit_app.py` (추론 로직 없음, API URL만 호출)

---

## 데이터셋

### NASA C-MAPSS (FD001)

- 출처: NASA Prognostics — 터보팬 엔진 **시뮬레이션** 데이터.
- **FD001**: 단일 운전 조건, 단일 고장 모드; 학습/테스트 유닛이 분리되어 제공되는 전형적인 RUL 벤치마크입니다.

### 주요 컬럼

| 구분 | 설명 |
|------|------|
| **unit_id** | 엔진(유닛) 식별자 |
| **cycle** | 운전 사이클(시간 인덱스) |
| **op_setting_1 ~ 3** | 운전 설정값 |
| **sensor_*** (일부는 학습에서 제외 가능) | 온도·압력 등 센서 측정값 |

학습 파이프라인에서는 **상수(분산 0) 컬럼 제거** 등으로 실제 입력 feature 개수는 데이터·스키마 기준으로 확정되며, **API는 학습 시 저장된 feature 이름 목록과 정확히 일치**해야 합니다.

---

## 방법론

### RUL 정의

- 유닛별 최대 cycle을 기준으로 `RUL = max_cycle - cycle` 형태로 잔여 사이클을 정의합니다.
- 과도하게 큰 RUL의 영향을 줄이기 위해 **상한(cap)** 을 둘 수 있습니다(학습 설정).

### Feature engineering

- 입력 후보: `cycle`, 운전 설정 3개, 센서 21개 등 **tabular feature**.
- 학습 데이터 기준 **분산이 0인(상수) 컬럼 제거** 후 스키마를 고정합니다.
- **스키마는 학습 데이터에서만 결정**하고, 검증·테스트·**추론 API**에서 동일한 컬럼 집합을 사용합니다.

### 모델 선택 (LightGBM)

- **LightGBM 회귀**를 기본으로 사용합니다(속도·tabular 성능·해석 가능성).
- 환경에 따라 **RandomForest** 등 대안 경로가 코드상 존재할 수 있습니다.

### 평가 지표 (RMSE, MAE)

- **RMSE**: 큰 오차에 민감한 전형적인 회귀 지표.
- **MAE**: 평균 절대 오차로 해석이 직관적입니다.
- 검증은 **unit_id 단위 분할**로 수행해 시계열·유닛 단위 **data leakage**를 피합니다. (같은 유닛이 train과 val에 동시에 섞이지 않도록 합니다.)

---

## 추론 및 위험도 분류

### 마지막 cycle 사용

- 시계열 전체가 아니라 **“현재 시점”** 을 대표하는 한 행(예: 유닛별 **마지막 cycle**)의 feature로 예측하는 패턴과 맞춥니다.
- **API**는 그와 같이 **한 시점의 feature dict** 한 벌을 `features`로 받습니다.

### SAFE / WARNING / CRITICAL 기준 (API: `classify_risk`)

기본 임계값(코드 상수, 변경 가능):

| 등급 | 조건 (예측 RUL 기준) |
|------|------------------------|
| **SAFE** | RUL > `warning_threshold` (**기본 30**) |
| **WARNING** | `critical_threshold` < RUL ≤ `warning_threshold` (**기본 10 < RUL ≤ 30**) |
| **CRITICAL** | RUL ≤ `critical_threshold` (**기본 10**) |

Streamlit 게이지·안내 문구는 **별도의 구간(예: 0–30 / 30–80 / 80–150)** 을 쓰어 UX를 보강할 수 있으며, **API 응답의 `risk`** 는 위 규칙을 따릅니다.

---

## API

### `GET /health`

- 응답 예: `{ "status": "ok" }`

### `POST /predict`

**Request**

```json
{
  "features": {
    "cycle": 100.0,
    "op_setting_1": 0.0,
    "op_setting_2": 0.0,
    "sensor_2": 518.67
  }
}
```

- 실제 키는 **학습 시 확정된 feature 목록 전체**여야 합니다.
- 누락·허용되지 않은 키·비숫자 → **422**  
- 모델 미로드 → **503**

**Response (성공)**

```json
{
  "prediction": 112.34,
  "risk": "SAFE"
}
```

배포 예시: Streamlit 기본값 `API_URL` — `https://predictive-maintenance-1-hoz8.onrender.com/predict` (`streamlit_app.py` 참고). 운영 시 실제 배포 URL과 맞출 것.

---

## 대시보드

### 입력 → 예측 → 시각화 흐름

1. 좌측에서 **feature**를 수정하고 **예측 요청**을 누릅니다.  
2. 우측에 **결과**가 표시됩니다(첫 방문 시 **placeholder** 안내).  
3. API 응답 `prediction`, `risk`를 기반으로 metric·메시지·차트를 갱신합니다.

### Gauge chart

- Plotly **indicator(gauge)** 로 RUL과 구간 색상을 표시합니다.

### RUL trend graph

- 사이클(X)–RUL(Y) **라인 차트**는 **시뮬레이션** 과거 추이(실제 설비 로그 아님, UI 캡션 안내).

### UX 개선 요소

- 좌·우 **2열** 레이아웃, 카드 스타일 **CSS**, 예측 전 **placeholder**, RUL 의미 **caption**, API `risk` 참고 표시 등.

---

## 배포

- **FastAPI**: 예) **Render**에 배포해 공개 엔드포인트로 제공.  
- **Streamlit**: 예) **Render**에서 Streamlit 앱으로 실행.  
- 저장소의 `streamlit_app.py` **`API_URL`** 이 실제 백엔드 주소와 일치해야 합니다.

---

## 설계 의도 (VERY IMPORTANT)

1. **게이지( Gauge )**  
   RUL은 숫자 하나보다 **어느 위험 구간에 있는지**가 의사결정에 직결됩니다. 구간 색·눈금으로 **한눈에 상태**를 읽게 했습니다.

2. **시각화 추가**  
   API만으로도 동작하지만, 데모·포트폴리오에서는 **맥락을 한 화면에** 담는 것이 중요합니다. 추이 그래프는 **시뮬레이션**으로 트렌드 이해를 돕고, 추후 실데이터로 교체 가능한 자리입니다.

3. **API + UI 분리**  
   추론 계약을 **HTTP API**로 고정해 Streamlit 외 **다른 클라이언트**도 동일하게 호출할 수 있습니다. UI는 **표현·입력**에 집중하고 배포·스케일을 분리합니다.

---

## 프로젝트 구조

저장소를 기준으로 한 디렉터리 요약입니다. `data/`, `output/` 일부는 **용량·라이선스·.gitignore** 때문에 클론 환경마다 유무가 다를 수 있습니다.

```text
predictive-maintenance/
├── README.md
├── requirements.txt
├── streamlit_app.py          # Streamlit 대시보드 (배포 API 호출)
├── app.py                     # 배포용 ASGI 진입점 (src.api.main 재노출, GET /)
├── scripts/
│   └── train_and_save.py      # 학습·아티팩트 저장 엔트리
├── doc/
│   └── requirements.md
├── data/
│   ├── raw/                   # FD001 등 원본 텍스트 (로컬 전용일 수 있음)
│   └── archive/               # NASA C-MAPSS 원본 아카이브·문서 (선택)
├── output/                    # 학습·평가 산출물
│   ├── model/
│   │   ├── model.joblib       # FastAPI가 로드하는 모델
│   │   └── feature_columns.json
│   ├── evaluation_results.txt
│   └── test_predictions*.csv
└── src/
    ├── api/
    │   ├── main.py            # FastAPI: /health, /predict
    │   └── artifact.py      # 모델·스키마 로드
    ├── data/
    │   ├── load.py
    │   └── rul.py
    ├── features/
    │   └── engineer.py
    ├── inference/
    │   ├── predict.py       # 배치 추론 등
    │   └── risk.py          # classify_risk
    └── models/
        ├── train.py
        └── evaluate.py
```

---

## 실행 방법

※ 아래는 대시보드 실행 예시입니다. API는 배포 URL을 쓰거나, 로컬에서 띄운 뒤 `streamlit_app.py`의 `API_URL`을 로컬로 바꿉니다.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

API만 로컬에서 띄울 때(모듈 경로는 환경에 맞게 조정):

```bash
uvicorn src.api.main:app --reload
```

---

## 핵심 학습 사항

- **Data leakage 방지**: 유닛이 겹치지 않도록 **unit_id 기준** train/validation 분할.
- **Train / Inference 일관성**: feature 목록·전처리(상수 제거 등)는 **학습에서 정한 스키마**를 추론·API에 그대로 적용.
- **서비스 구현 경험**: 모델을 **REST API**로 노출하고, **Streamlit**으로 비기술 이해관계자도 볼 수 있는 화면을 분리 구현.

---

## 향후 개선

- **실시간·스트리밍** 센서 데이터와 API 연동(“마지막 시점” 정의 일관화).
- **Feature 확장**: 롤링 통계·추세 feature 등(학습·추론 스키마 동시 유지).
- 다중 조건 데이터(FD002 등)로 **파이프라인 재사용** 검증.

---

## 참고·라이선스

- NASA C-MAPSS 사용 시 **데이터셋 라이선스·출처** 표기를 따릅니다.  
- Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008.
