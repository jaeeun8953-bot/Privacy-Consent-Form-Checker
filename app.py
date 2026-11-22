"""Pilot evaluation of staged similarity and GPT-5."""
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError

import altair as alt
import pandas as pd
import streamlit as st

from model_study import STUDY_MODEL, measure, summarize
from similarity import BLACK, WHITE, classify, reference_vectors, score

THRESHOLDS = (20, 40, 60, 80)
SIMILARITY = "TF-IDF + 코사인 유사도"
GPT = "GPT-5"


def measure_similarity(sentence, percent):
    start = perf_counter()
    white, black, white_match, black_match = score([sentence])[0]
    prediction = classify(white, black, percent / 100, percent / 100)
    return {
        "유사도": prediction, "유사도 시간(초)": perf_counter() - start,
        "화이트 최고": white, "블랙 최고": black,
        "화이트 근접 표현": white_match, "블랙 근접 표현": black_match,
    }


def metrics(frame, method):
    pred = frame[method]
    truth = frame["정답"]
    counts = summarize([{"정답": row["정답"], "예측": row[method], "응답 시간(초)": row[f"{method} 시간(초)"]}
                        for row in frame.to_dict("records")])
    counts["미탐"] += int(((truth == "위험") & (pred == "판단 유보")).sum())
    counts["유보"] = int((pred == "판단 유보").sum())
    return counts


st.set_page_config(page_title="개인정보 동의서 후속연구", layout="wide")
st.title("개인정보 동의서 위험 문장 검증")
st.caption("새 가상 문장 200개에서 화이트리스트 → 블랙리스트 기반 TF-IDF·코사인 유사도와 GPT-5를 비교해요.")
data = Path(__file__).with_name("pilot_samples_200.csv").read_bytes()
examples = list(csv.DictReader(StringIO(data.decode("utf-8-sig"), newline="")))
st.download_button("검증 데이터 200개 CSV", data, "pilot_samples_200.csv", mime="text/csv")
with st.expander("검증 데이터와 기준 목록"):
    st.write("10개 유형별로 안전 10개와 위험 10개를 만들었어요. 유형마다 안전·위험 표현 5쌍을 만들고 온라인 신청과 방문 접수에 적용했어요.")
    st.dataframe(pd.DataFrame(examples).groupby(["category", "label"]).size().unstack().reset_index().rename(columns={"category": "유형"}), hide_index=True)
    a, b = st.columns(2)
    a.write("화이트리스트 · 100개")
    a.dataframe(pd.DataFrame({"기준 표현": WHITE}), hide_index=True)
    b.write("블랙리스트 · 100개")
    b.dataframe(pd.DataFrame({"기준 표현": BLACK}), hide_index=True)

session_key = (sha256(data).hexdigest(), STUDY_MODEL)
if st.session_state.get("pilot_source") != session_key:
    st.session_state.pilot_source = session_key
    st.session_state.pilot_results = []
results = st.session_state.pilot_results
percent = st.selectbox("유사도 임계값", THRESHOLDS, format_func=lambda value: f"{value}% (화이트·블랙 공통)")
st.caption("임계값을 바꾸면 같은 GPT 응답으로 유사도 결과만 다시 계산해요. 전체 점수를 보고 임계값을 고를 경우 탐색 결과로 해석하세요.")
progress = st.progress(len(results) / len(examples), text=f"{len(results)}/{len(examples)}문장 검증 완료")
if len(results) < len(examples):
    batch = st.selectbox("한 번에 검증할 문장 수", [5, 10, 20], index=1)
    count = min(batch, len(examples) - len(results))
    st.caption("가상 문장만 문장당 한 번 API에 보내요. 정답 라벨은 앱에서 채점하며 API 이용료가 발생할 수 있어요.")
    if st.button(f"다음 {count}문장 검증", type="primary"):
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except FileNotFoundError:
            key = ""
        if not key:
            st.warning("Streamlit Secrets의 OPENAI_API_KEY를 확인해 주세요.")
        else:
            try:
                for sample in examples[len(results):len(results) + count]:
                    result = measure(sample, key)
                    results.append({"원문": result["원문"], "정답": result["정답"], "GPT": result["예측"],
                                    "GPT 시간(초)": result["응답 시간(초)"], "응답 모델": result["모델"],
                                    "유형": sample["category"], "문장 쌍": sample["pair_id"]})
                    progress.progress(len(results) / len(examples), text=f"{len(results)}/{len(examples)}문장 검증 완료")
            except (HTTPError, OSError, ValueError, KeyError) as error:
                st.error(f"검증이 중단됐어요 ({getattr(error, 'code', type(error).__name__)}). 남은 문장부터 이어갈 수 있어요.")

st.subheader("정확도와 문장당 평균 처리 시간")
if not results:
    st.info("첫 검증을 실행하면 두 방식의 실측 결과가 표시돼요.")
    st.stop()
reference_vectors()
table = pd.DataFrame([{**row, **measure_similarity(row["원문"], percent)} for row in results])
summary = pd.DataFrame([{"방식": label, **metrics(table, method)}
                        for label, method in [(SIMILARITY, "유사도"), (GPT, "GPT")]])
st.caption(f"두 방식이 모두 처리한 {len(table)}문장 기준 · 응답 모델 {results[-1]['응답 모델']}")
st.caption("유사도는 기준 벡터 준비가 끝난 뒤 문장 하나를 벡터화하고 분류하는 시간을 재요. GPT-5는 API 왕복 시간을 재므로 네트워크 시간도 포함돼요.")
a, b = st.columns(2)
for column, metric, unit in [(a, "정확도", "%"), (b, "문장당 시간(초)", "초")]:
    chart = alt.Chart(summary).mark_line(point=True).encode(
        x=alt.X("방식:N", sort=[SIMILARITY, GPT], axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{metric}:Q", title=f"{metric} ({unit})",
                scale=alt.Scale(domain=[0, 100]) if unit == "%" else alt.Scale(zero=True)),
        tooltip=["방식:N", alt.Tooltip(f"{metric}:Q", format=".4f")],
    )
    column.altair_chart(chart.properties(height=230), use_container_width=True)
st.dataframe(summary, hide_index=True)
st.caption("판단 유보는 정확도에서 오답으로 집계해요. 위험 문장을 유보하면 미탐에도 포함해요.")
st.subheader("유형별 결과")
st.dataframe(pd.DataFrame([
    {"유형": category, "검증 문장": len(group), "방식": label, **metrics(group, method)}
    for category, group in table.groupby("유형", sort=False)
    for label, method in [(SIMILARITY, "유사도"), (GPT, "GPT")]
]), hide_index=True)
st.subheader("판단이 갈린 문장과 오탐·미탐")
different, same, errors = st.tabs(["다르게 판단", "같게 판단", "오탐·미탐·유보"])
with different:
    st.dataframe(table[table["유사도"] != table["GPT"]], hide_index=True)
with same:
    st.dataframe(table[table["유사도"] == table["GPT"]], hide_index=True)
with errors:
    st.dataframe(table[(table["정답"] != table["유사도"]) | (table["정답"] != table["GPT"])], hide_index=True)
st.download_button("문장별 결과 CSV 저장", table.to_csv(index=False).encode("utf-8-sig"), "pilot-results.csv", mime="text/csv")
