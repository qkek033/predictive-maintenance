"""
Streamlit UI: 배포된 FastAPI /predict 엔드포인트에 feature dict를 POST하고
예측 RUL·위험도를 표시한다. 백엔드 코드는 수정하지 않음.
"""
import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

# 배포 API (경로는 /predict)
API_URL = "https://predictive-maintenance-1-hoz8.onrender.com/predict"

# 환경변수로 타임아웃 설정. 배포/네트워크별로 조정 가능하며 기본 30초.
# (설정 예: API_TIMEOUT=60)
TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# 입력 필드 순서 및 테스트용 기본값 (CMAPSS FD001 스케일에 가깝게)
FEATURE_DEFAULTS: dict[str, float] = {
    "cycle": 100.0,
    "op_setting_1": 0.0,
    "op_setting_2": 0.0,
    "sensor_2": 518.67,
    "sensor_3": 642.15,
    "sensor_4": 1589.70,
    "sensor_6": 14.62,
    "sensor_7": 21.61,
    "sensor_8": 553.75,
    "sensor_9": 2388.04,
    "sensor_11": 47.49,
    "sensor_12": 522.28,
    "sensor_13": 2388.07,
    "sensor_14": 8131.49,
    "sensor_15": 8.43,
    "sensor_17": 392.0,
    "sensor_20": 39.0,
    "sensor_21": 23.42,
}

# Plotly 한글 렌더링: 한글 글리프가 있는 폰트를 우선 지정 (NanumGothic, Malgun Gothic)
_PLOTLY_FONT_KO = "Arial, NanumGothic, Malgun Gothic"


def post_predict(url: str, features: dict, timeout: int = TIMEOUT) -> dict:
    """POST JSON {"features": {...}} 후 응답 dict 반환. timeout은 환경변수 API_TIMEOUT 반영."""
    payload = json.dumps({"features": features}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _risk_color_html(risk: str) -> str:
    """위험도별 색상: SAFE=녹색, WARNING=주황, CRITICAL=빨강. st.markdown용 HTML 반환."""
    color = {"SAFE": "green", "WARNING": "orange", "CRITICAL": "red"}.get(risk, "gray")
    return f"<h3 style='color: {color};'>{risk}</h3>"


# RUL 게이지·메시지 공통 임계값 (실무 의사결정용 구간 정의)
RUL_CRITICAL_MAX = 30.0  # 0~30: CRITICAL
RUL_WARNING_MAX = 80.0  # 30~80: WARNING
RUL_SAFE_MAX = 150.0  # 80~150: SAFE (게이지 기본 상한)


def _rul_zone_from_value(v: float) -> str:
    """예측 RUL로 게이지 구간 키 반환: CRITICAL | WARNING | SAFE."""
    if v < RUL_CRITICAL_MAX:
        return "CRITICAL"
    if v < RUL_WARNING_MAX:
        return "WARNING"
    return "SAFE"


def _rul_status_message(zone: str) -> str:
    """구간별 운영자용 설명 문구 (게이지 해석과 동일 기준)."""
    return {
        "SAFE": "안전 상태 (충분한 잔여 수명 확보)",
        "WARNING": "주의 상태 (점검 또는 예방 정비 필요)",
        "CRITICAL": "위험 상태 (즉시 유지보수 필요)",
    }.get(zone, "상태를 확인할 수 없습니다.")


def _rul_failure_horizon_text(v: float) -> str:
    """잔여 수명을 사람이 읽기 쉬운 한 줄로 표현."""
    if v < 0:
        v = 0.0
    # 소수는 반올림해 정수 사이클로 안내 (과도한 소수 자릿수 방지)
    n = int(round(v))
    return f"약 {n} 사이클 후 고장 예상"


def _gauge_fig(value: float):
    """
    RUL Plotly 게이지.
    - 구간: 0~30 CRITICAL, 30~80 WARNING, 80~150 SAFE (색 + 눈금)
    - 임계값 30·80·150 시각 표시 및 주석으로 의미 설명
    """
    import plotly.graph_objects as go

    v = float(value)
    # 게이지 축은 150 초과 예측도 바늘로 볼 수 있게 상한 확장
    axis_hi = max(RUL_SAFE_MAX, v)
    needle = min(max(v, 0.0), axis_hi)

    gauge_steps: list[dict] = [
        {"range": [0, RUL_CRITICAL_MAX], "color": "rgba(220, 38, 38, 0.45)"},
        {"range": [RUL_CRITICAL_MAX, RUL_WARNING_MAX], "color": "rgba(234, 179, 8, 0.45)"},
        {"range": [RUL_WARNING_MAX, RUL_SAFE_MAX], "color": "rgba(22, 163, 74, 0.45)"},
    ]
    # 150 사이클 초과 예측: 표준 SAFE 구간 밖은 연한 회색으로 표시
    if axis_hi > RUL_SAFE_MAX:
        gauge_steps.append({"range": [RUL_SAFE_MAX, axis_hi], "color": "rgba(148, 163, 184, 0.35)"})

    tickvals = [0.0, RUL_CRITICAL_MAX, RUL_WARNING_MAX, RUL_SAFE_MAX]
    ticktext = ["0", "30", "80", "150"]
    if axis_hi > RUL_SAFE_MAX:
        tickvals.append(axis_hi)
        ticktext.append(f"{axis_hi:.0f}")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=needle,
            number={
                "font": {"size": 30, "color": "white", "family": _PLOTLY_FONT_KO},
                "suffix": " 사이클",
                "valueformat": ".1f",
            },
            domain={"x": [0.08, 0.92], "y": [0.18, 0.95]},
            # 제목은 layout.title로 분리해 게이지와 겹침 방지
            title={"text": ""},
            gauge={
                "axis": {
                    "range": [0, axis_hi],
                    "tickmode": "array",
                    "tickvals": tickvals,
                    "ticktext": ticktext,
                    "tickfont": {"color": "white", "family": _PLOTLY_FONT_KO},
                },
                "bar": {"color": "rgb(30, 64, 175)"},
                "steps": gauge_steps,
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.82,
                    "value": needle,
                },
            },
        )
    )

    # 임계 구간 설명 (게이지 하단, 운영 기준을 한눈에)
    fig.add_annotation(
        x=0.5,
        y=0.02,
        xref="paper",
        yref="paper",
        text=(
            "<b>구간 기준</b> · 0~30: CRITICAL(위험) · "
            "30~80: WARNING(주의) · 80~150: SAFE(안전)"
        ),
        showarrow=False,
        font={"size": 12, "color": "white", "family": _PLOTLY_FONT_KO},
        align="center",
    )

    fig.update_layout(
        font=dict(family=_PLOTLY_FONT_KO, color="white"),
        title={
            "text": (
                "예측 잔여 수명 (RUL)<br>"
                #"<sup>값이 클수록 고장까지 여유가 있음 (사이클 단위)</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": dict(family=_PLOTLY_FONT_KO, color="white"),
        },
        xaxis=dict(title=dict(font=dict(family=_PLOTLY_FONT_KO, color="white"))),
        yaxis=dict(title=dict(font=dict(family=_PLOTLY_FONT_KO, color="white"))),
        margin=dict(l=24, r=24, t=80, b=72),
        autosize=False,
        width=500,
        height=300,
        paper_bgcolor="#2b2b2b",
        plot_bgcolor="#2b2b2b",
    )
    return fig


def _simulated_rul_trend_df(end_cycle: float, end_rul: float) -> pd.DataFrame:
    """
    과거 RUL 감소 추세를 보여주기 위한 시뮬레이션 데이터.
    실제 장비 로그·API 이력이 아니며, 현재 입력 사이클·예측 RUL에 맞춰 생성한다.
    """
    ec = float(end_cycle)
    er = max(float(end_rul), 0.0)
    n = 16
    start_c = max(1.0, ec - (n - 1))
    denom = max(n - 1, 1)
    cycles = [start_c + (ec - start_c) * i / denom for i in range(n)]
    start_rul = min(er + 40.0, 150.0)
    if start_rul <= er:
        start_rul = min(er + 5.0, 150.0)
    ruls = [start_rul - (start_rul - er) * i / denom for i in range(n)]
    return pd.DataFrame({"cycle": cycles, "RUL": ruls})


def main():
    st.set_page_config(page_title="예지보전 RUL 예측", layout="wide")
    if "show_result" not in st.session_state:
        st.session_state.show_result = False
    st.title("예지보전 RUL 예측")
    st.caption("배포 API에 feature를 전송하고 예측·위험도를 확인합니다.")

    st.markdown(
        """
<style>
/* 최상위 좌·우 패널만 (HB > column 직계). :first/:last만 쓰면 안쪽 입력 2열에도 테두리가 중복됨 */
[data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child > div,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child > div,
div.stHorizontalBlock > div[data-testid="column"]:first-child > div,
div.stHorizontalBlock > div[data-testid="stColumn"]:first-child > div {
    border: 3px solid #3b82f6;
    border-radius: 12px;
    padding: 20px;
    background-color: #1e1e1e;
    margin-right: 10px;
    box-sizing: border-box;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child > div,
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child > div,
div.stHorizontalBlock > div[data-testid="column"]:last-child > div,
div.stHorizontalBlock > div[data-testid="stColumn"]:last-child > div {
    border: 3px solid #22c55e;
    border-radius: 12px;
    padding: 20px;
    background-color: #1e1e1e;
    margin-left: 10px;
    box-sizing: border-box;
}

/* 안쪽 입력용 HB의 직계 column 래퍼: 개별 테두리 제거 → 한 카드로만 보이게 */
[data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] > [data-testid="column"] > div,
[data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div,
div.stHorizontalBlock div.stHorizontalBlock > div[data-testid="column"] > div,
div.stHorizontalBlock div.stHorizontalBlock > div[data-testid="stColumn"] > div {
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border-radius: 0 !important;
}

/* 입력 feature 2열(stHorizontalBlock) 전체를 하나의 카드로 */
[data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"],
div.stHorizontalBlock div.stHorizontalBlock {
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 16px;
    background-color: #262626;
    box-sizing: border-box;
}
</style>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("입력 feature")
        values: dict[str, float] = {}
        cols = st.columns(2)
        keys = list(FEATURE_DEFAULTS.keys())
        for i, name in enumerate(keys):
            with cols[i % 2]:
                values[name] = st.number_input(
                    label=name,
                    value=float(FEATURE_DEFAULTS[name]),
                    format="%.6f",
                    key=f"in_{name}",
                )

        predict_clicked = st.button("예측 요청", type="primary")
        if predict_clicked:
            st.session_state.show_result = True

    with right:
        if not st.session_state.show_result:
            st.markdown(
                """
<div style="
    min-height: 55vh;
    width: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: #9ca3af;
    text-align: center;
    font-size: 25px;
    line-height: 1.6;
    box-sizing: border-box;
">
    <div style="font-size: 40px; line-height: 1;"></div>
    <div style="margin-top: 10px; max-width: 320px;">
        설비 데이터를 입력하고 <br>예측을 실행하면<br>
        <b>잔여 수명(RUL)</b>과 <b>위험도</b>가 이 영역에 표시됩니다.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            if predict_clicked:
                try:
                    result = post_predict(API_URL, values)
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", errors="replace")
                    st.error(f"API 오류 ({e.code}): {body}")
                    return
                except urllib.error.URLError as e:
                    st.error(f"연결 실패: {e.reason}")
                    return
                except Exception as e:
                    st.error(f"요청 처리 실패: {e}")
                    return

                prediction = result.get("prediction")
                risk = result.get("risk", "UNKNOWN")

                st.subheader("결과")

                if prediction is None:
                    st.error("Prediction failed")
                else:
                    try:
                        pred_float = float(prediction)
                    except (TypeError, ValueError):
                        st.error("Prediction failed")
                    else:
                        st.metric(label="예측 RUL (사이클)", value=f"{pred_float:.4f}")

                        rul_zone = _rul_zone_from_value(pred_float)
                        st.markdown(_risk_color_html(rul_zone), unsafe_allow_html=True)
                        msg = _rul_status_message(rul_zone)
                        if rul_zone == "SAFE":
                            st.success(msg)
                        elif rul_zone == "WARNING":
                            st.warning(msg)
                        else:
                            st.error(msg)

                        st.caption(
                            f"{_rul_failure_horizon_text(pred_float)} · 서버 위험도(risk): {risk}"
                        )

                        st.plotly_chart(_gauge_fig(pred_float), use_container_width=False)

                st.subheader("RUL 변화 추이")
                trend_pred: float | None = None
                if prediction is not None:
                    try:
                        trend_pred = float(prediction)
                    except (TypeError, ValueError):
                        trend_pred = None
                if trend_pred is not None:
                    st.caption(
                        "※ 본 그래프는 시뮬레이션된 과거 RUL 추이입니다. 실제 설비 이력 데이터가 아닙니다."
                    )
                    trend_df = _simulated_rul_trend_df(values["cycle"], trend_pred)
                    st.line_chart(trend_df, x="cycle", y="RUL")
                    st.caption(
                        "RUL은 Remaining Useful Life로, 값이 낮을수록 고장에 가까움을 의미합니다."
                    )
                else:
                    st.caption("예측값이 있을 때 사이클 대비 RUL 추이(시뮬레이션)를 표시합니다.")


if __name__ == "__main__":
    main()
