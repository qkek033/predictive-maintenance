# 요구사항 명세서 (Requirements Specification)

---

## 1. 목적 (Purpose)

본 프로젝트는 NASA CMAPSS turbofan 데이터(FD001)를 활용하여  
센서 기반 장비 상태를 분석하고 Remaining Useful Life(RUL)를 예측하는  
예지보전(Predictive Maintenance) 시스템을 구축하는 것을 목적으로 한다.

또한 본 시스템은 다음과 같은 실무적 가치를 제공하는 것을 목표로 한다:

- 장비 고장을 사전에 예측하여 유지보수 비용 절감
- 불필요한 정비를 줄이고 다운타임 최소화
- 데이터 기반 운영 의사결정 지원

---

## 2. 범위 (Scope)

- 데이터셋: NASA CMAPSS (FD001)
- 모델: Tabular 기반 회귀 모델
  - 기본: LightGBM
  - 대체: RandomForest
- 시스템 구성:
  - 데이터 전처리 및 RUL 생성
  - Feature Engineering
  - 모델 학습 및 평가
  - 테스트 데이터 추론
  - 위험도 분류 로직
  - FastAPI 기반 추론 API
  - Streamlit 기반 UI

---

## 3. 기능 요구사항 (Functional Requirements)

---

### FR-01 데이터 로딩

- `data/raw/` 경로에서 `train_FD001.txt`, `test_FD001.txt`를 로드해야 한다
- 공백(whitespace) 기반 구분자를 사용하여 파싱해야 한다
- 컬럼 구성:
  - `unit_id`, `cycle`
  - `op_setting_1` ~ `op_setting_3`
  - `sensor_1` ~ `sensor_21`
- 불필요한 trailing empty column 제거

---

### FR-02 RUL 생성

- 각 `unit_id`별 최대 cycle을 계산해야 한다
- RUL 정의:
  - `RUL = max_cycle - cycle`
- RUL upper bound cap 적용 가능해야 함 (configurable)
- RUL 분포 확인 및 이상 값 점검 수행

---

### FR-03 Feature Engineering

- 입력 feature 구성:
  - cycle
  - operational settings
  - sensor values
- 상수 컬럼 및 불필요 컬럼 제거
- train / test 간 동일한 feature schema 유지
- feature 생성 로직은 재사용 가능하도록 구성

---

### FR-04 모델 학습

- RUL 예측을 위한 회귀 모델 학습 가능해야 한다
- 모델:
  - 기본: LightGBM
  - fallback: RandomForest
- 데이터 분할:
  - `unit_id` 기준 split (data leakage 방지)
- 재현 가능한 학습을 위해 random seed 고정

---

### FR-05 모델 평가

- 평가 지표:
  - RMSE
  - MAE
- feature importance 출력 가능해야 한다
- 모델 성능 결과를 로그 또는 파일로 저장

---

### FR-06 테스트 추론

- test 데이터에서 각 `unit_id`의 마지막 cycle만 사용해야 한다
- RUL 예측 수행
- `RUL_FD001.txt`와 비교하여 평가 가능해야 한다
- 예측 결과 저장 (CSV)

---

### FR-07 위험도 판단

- RUL 기반 상태 분류:
  - SAFE
  - WARNING
  - CRITICAL
- threshold는 설정값으로 관리되어야 한다

- 운영 기준 정의:
  - SAFE: 정상 운영
  - WARNING: 점검 필요 (예방 정비 권장)
  - CRITICAL: 즉시 유지보수 필요

---

### FR-08 추론 API

- `GET /health`
  - 서비스 상태 반환
- `POST /predict`
  - 입력: feature 값
  - 출력:
    - predicted_rul
    - risk_status

- 입력 데이터 검증 수행:
  - 필수 feature 누락 시 에러 처리
  - 잘못된 값 타입 처리

---

### FR-09 UI (Streamlit)

- 센서값 입력 인터페이스 제공
- 예측 결과 시각화:
  - RUL
  - 위험도
- 결과 해석을 위한 간단한 설명 제공

---

## 4. 비기능 요구사항 (Non-Functional Requirements)

---

### NFR-01 재현성

- 동일한 코드 실행 시 동일한 결과를 재현할 수 있어야 한다

---

### NFR-02 모듈화

- 코드 구조는 다음과 같이 분리되어야 한다:
  - data
  - features
  - models
  - inference

---

### NFR-03 단순성

- 과도한 구조 설계 지양
- 유지보수 가능한 수준 유지

---

### NFR-04 확장성

- FD002 ~ FD004 데이터셋으로 확장 가능해야 한다

---

### NFR-05 일관성

- train과 inference의 feature 생성 로직은 동일해야 한다

---

### NFR-06 데이터 품질 대응

- 결측값 및 이상치 처리 고려
- 잘못된 입력 데이터에 대한 방어 로직 포함

---

### NFR-07 로깅 및 관측성

- 주요 처리 단계에서 로그 출력
- 모델 입력 및 출력 추적 가능

---

## 5. 환경 요구사항 (Environment Requirements)

| 항목 | 내용 |
|------|------|
| Python | 3.10 이상 |
| 주요 라이브러리 | pandas, numpy, scikit-learn |
| 모델 | lightgbm (선택) |
| API | fastapi, uvicorn |
| UI | streamlit |

---

## 6. 수용 기준 (Acceptance Criteria)

- [ ] train 데이터에서 RUL 생성이 정상 동작
- [ ] feature dataset 생성 완료
- [ ] 모델 학습 및 RMSE 출력 가능
- [ ] test 데이터 예측 결과 생성
- [ ] 위험도 분류 로직 정상 동작
- [ ] `/health` endpoint 정상 응답
- [ ] `/predict` endpoint 정상 응답
- [ ] Streamlit UI에서 예측 결과 확인 가능

---