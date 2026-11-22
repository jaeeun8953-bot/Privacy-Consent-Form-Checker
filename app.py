"""GPT-3 historical baseline and current OpenAI model follow-up."""
from pathlib import Path
from hashlib import sha256
from urllib.error import HTTPError

import altair as alt
import pandas as pd
import streamlit as st

from model_study import PROMPT, measure, parse_csv, summarize
from run_review import MODEL

st.set_page_config(page_title="개인정보 동의서 후속연구", layout="wide")
st.title("개인정보 동의서 위험 문장 검출 후속연구")
st.markdown("**연구 질문** · 후속 모델이 기존 GPT-3보다 정확하게 판단하고 더 빠르게 응답하는가?")
st.caption("논문에서 제시한 예상을 정확도와 문장당 평균 응답 시간으로 검증해요.")
st.subheader("기존 논문 기록")
st.dataframe([
    {"평가 조건": "기존 검증", "GPT-3 정확도(%)": 76.05},
    {"평가 조건": "신규 유형 검증", "GPT-3 정확도(%)": 72.65},
], hide_index=True)
st.caption("기존 GPT-3 기록의 문장당 평균 시간은 59초예요. 이 화면에서는 후속 모델의 API 요청부터 응답 수신까지 측정해요.")
st.subheader("후속 모델 검증")
st.caption(f"설정 모델 · {MODEL}")
data = Path(__file__).with_name("followup_samples_2000.csv").read_bytes()
examples = parse_csv(data)
GROUP_SIZE = 200
CATEGORIES = [('개인정보 수집', '목적에 필요한 정보만 받는지'), ('마케팅 동의', '광고 수신 선택이 기본 서비스와 분리되는지'), ('보유·파기', '보유 기간과 파기 시점이 구체적인지'), ('동의 거부', '선택 정보 제공을 거부해도 접수할 수 있는지'), ('동의 방식', '직접 동의하는지, 미응답을 동의로 간주하는지'), ('제3자 제공', '제공받는 기관을 확인할 수 있는지'), ('이용 목적', '고지한 목적 범위에서만 사용하는지'), ('아동 동의', '만 14세 미만 아동의 법정대리인 동의를 확인하는지'), ('민감정보', '건강정보 등 민감정보 동의를 구분하는지'), ('고유식별정보', '여권번호 등 고유식별정보 동의를 구분하는지')]
st.caption("가상 문장 2,000개 · 분야 10개 · 각 분야는 20개 상황 × 5개 신청 경로 × 안전·위험 2개 라벨로 구성했어요.")
st.download_button("새 검증 데이터 CSV 받기", data, "followup_samples_2000.csv", mime="text/csv")

with st.expander("후속 모델에 전달하는 분류 지시"):
    instructions = st.text_area("분류 지시", PROMPT, help="원래 연구의 분류 기준에 맞춰 조정할 수 있어요.")
fingerprint = (sha256(data).hexdigest(), MODEL, instructions)
if st.session_state.get("followup_source") != fingerprint:
    st.session_state.followup_source = fingerprint
    st.session_state.followup_results = []
results = st.session_state.followup_results
st.progress(len(results) / len(examples), text=f"검증 {len(results):,}/{len(examples):,}문장")
st.caption("문장당 API 요청을 한 번 보내고, 정답은 앱에서 채점해요. 요청 비용이 발생할 수 있어요.")

if len(results) < len(examples):
    batch = st.selectbox("한 번에 검증할 문장 수", [5, 10, 20, 50], index=1)
    count = min(batch, GROUP_SIZE - len(results) % GROUP_SIZE, len(examples) - len(results))
    if st.button(f"다음 {count}문장 검증", type="primary"):
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except FileNotFoundError:
            key = ""
        if not key:
            st.warning("Streamlit Secrets의 OPENAI_API_KEY를 확인해 주세요.")
        else:
            status = st.empty()
            try:
                for row in examples[len(results): len(results) + count]:
                    status.caption(f"{len(results) + 1:,}번째 문장 검증 중…")
                    results.append(measure(row, key, instructions))
            except HTTPError as error:
                st.error(f"OpenAI API 요청 실패 ({error.code}). 완료한 문장까지 저장했어요.")
            except (OSError, ValueError, KeyError):
                st.error("응답을 채점하지 못했어요. 완료한 문장까지 저장했어요.")
            else:
                st.rerun()
            finally:
                status.empty()

st.subheader("분야별 정확도와 평균 응답 시간")
st.caption("X축은 각 분야의 마지막 문장 번호예요. 200문장 검증이 끝날 때마다 해당 분야 자체의 정확도와 평균 응답 시간이 한 점씩 추가돼요.")
st.caption("점선은 기존 GPT-3 논문 전체 기록이에요. 평가 데이터와 실행 환경이 달라 분야별 후속 모델 결과와 참고용으로 비교해요.")
ends = list(range(GROUP_SIZE, len(examples) + 1, GROUP_SIZE))
records = []
for end, (category, _) in zip(ends, CATEGORIES, strict=True):
    if end <= len(results):
        measured = summarize(results[end - GROUP_SIZE:end])
        records.append({"검증 구간 끝": end, "분야": category, **measured})
charts = st.columns(2)
for column, metric, axis, references in [
    (charts[0], "정확도", "정확도 (%)", [("GPT-3 기존 검증", 76.05), ("GPT-3 신규 유형 검증", 72.65)]),
    (charts[1], "문장당 시간(초)", "문장당 평균 응답 시간 (초)", [("GPT-3 평균", 59)]),
]:
    data_table = pd.DataFrame(records, columns=["검증 구간 끝", "분야", "정확도", "문장당 시간(초)", "오탐", "미탐"])
    y = alt.Y(f"{metric}:Q", title=axis, scale=alt.Scale(domain=[0, 100]) if metric == "정확도" else alt.Scale(zero=True))
    measured_chart = alt.Chart(data_table).mark_line(point=True, color="#235aa6").encode(
        x=alt.X("검증 구간 끝:Q", title="문장 번호", scale=alt.Scale(domain=[200, 2000], nice=False),
                axis=alt.Axis(values=ends, format="d", labelOverlap=False, labelAngle=-45)),
        y=y,
        tooltip=["분야:N", "검증 구간 끝:Q", alt.Tooltip(f"{metric}:Q", format=".3f")],
    )
    baseline = pd.DataFrame([{"기준": name, metric: value} for name, value in references])
    reference_chart = alt.Chart(baseline).mark_rule(strokeDash=[5, 4]).encode(
        y=alt.Y(f"{metric}:Q", axis=None), color=alt.Color("기준:N", title=None), tooltip=["기준:N", f"{metric}:Q"],
    )
    column.altair_chart((measured_chart + reference_chart).properties(title=axis, height=250), use_container_width=True)

with st.expander("분야를 나눈 기준", expanded=True):
    st.dataframe([{"문장 번호": f"{end - GROUP_SIZE + 1:,}–{end:,}", "분야": category,
                   "판정 기준": criterion, "구성": "안전 100 · 위험 100"}
                  for end, (category, criterion) in zip(ends, CATEGORIES, strict=True)], hide_index=True)

if results:
    stats = summarize(results)
    st.subheader(f"누적 실측 · {len(results):,}/{len(examples):,}문장")
    st.caption(f"응답 모델 · {results[-1]['모델']}")
    columns = st.columns(4)
    for column, (name, value) in zip(columns, stats.items()):
        column.metric(name, f"{value:.2f}%" if name == "정확도" else (f"{value:.3f}" if name.endswith("(초)") else value))
    with st.expander("문장별 결과와 오탐·미탐"):
        table = pd.DataFrame(results)
        choices = st.radio("보기", ["전체", "오탐", "미탐"], horizontal=True)
        filtered = table
        if choices != "전체":
            truth, prediction = ("안전", "위험") if choices == "오탐" else ("위험", "안전")
            filtered = table[(table["정답"] == truth) & (table["예측"] == prediction)]
        st.dataframe(filtered, hide_index=True)
        st.download_button("전체 결과 CSV 저장", table.to_csv(index=False).encode("utf-8-sig"),
                           "model-study-results.csv", mime="text/csv")
