"""GPT-3 historical baseline and current OpenAI model follow-up."""
from pathlib import Path
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
st.caption("새 가상 문장 2,000개 · 안전 1,000개, 위험 1,000개 · 20개 상황과 10개 위험 유형으로 구성")
st.download_button("새 검증 데이터 CSV 받기", data, "followup_samples_2000.csv", mime="text/csv")

with st.expander("후속 모델에 전달하는 분류 지시"):
    instructions = st.text_area("분류 지시", PROMPT, help="원래 연구의 분류 기준에 맞춰 조정할 수 있어요.")
fingerprint = ("synthetic-followup-2000-v1", MODEL, instructions)
if st.session_state.get("followup_source") != fingerprint:
    st.session_state.followup_source = fingerprint
    st.session_state.followup_results = []
results = st.session_state.followup_results
st.progress(len(results) / len(examples), text=f"검증 {len(results):,}/{len(examples):,}문장")
st.caption("문장당 API 요청을 한 번 보내고, 정답은 앱에서 채점해요. 요청 비용이 발생할 수 있어요.")

if len(results) < len(examples):
    batch = st.selectbox("한 번에 검증할 문장 수", [5, 10, 20, 50], index=1)
    count = min(batch, len(examples) - len(results))
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

if results:
    model = results[-1]["모델"]
    stats = summarize(results)
    st.subheader(f"실측 결과 · {len(results):,}/{len(examples):,}문장")
    st.caption("현재까지 완료한 문장의 누적 결과예요. 미완료 요청은 집계에서 제외해요.")
    columns = st.columns(4)
    for column, (name, value) in zip(columns, stats.items()):
        column.metric(name, f"{value:.2f}%" if name == "정확도" else (f"{value:.3f}" if name.endswith("(초)") else value))
    if len(results) == len(examples):
        st.caption("GPT-3는 당시 논문 기록, 후속 모델은 새 가상 문장 2,000개에서의 실측이에요. 평가 데이터와 API·네트워크 환경이 달라 모델 자체의 개선 폭을 직접 추정하기는 어려워요.")
        a, b = st.columns(2)
        for column, metric, old, current in [
            (a, "정확도 (%)", [76.05, 72.65], stats["정확도"]),
            (b, "문장당 평균 응답 시간 (초)", [59], stats["문장당 시간(초)"]),
        ]:
            baseline_names = ["GPT-3 · 기존 검증", "GPT-3 · 신규 유형 검증"] if len(old) == 2 else ["GPT-3 · 논문 기록"]
            values = pd.DataFrame({"측정": [*baseline_names, f"{model} · 새 데이터"], metric: [*old, current]})
            column.altair_chart(alt.Chart(values).mark_bar().encode(
                x=alt.X("측정:N", title=None),
                y=alt.Y(f"{metric}:Q"),
                color=alt.Color("측정:N", legend=None),
                tooltip=["측정:N", alt.Tooltip(f"{metric}:Q", format=".3f")],
            ), use_container_width=True)
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
