"""Pilot evaluation of staged similarity and GPT-5."""
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.error import HTTPError

import altair as alt
import pandas as pd
import streamlit as st

from model_study import STUDY_MODEL, measure, summarize
from similarity import BLACK, WHITE, classify, reference_vectors, score

BATCH_SIZE = 20
THRESHOLDS = (20, 40, 60, 80)
SIMILARITY = "TF-IDF + 코사인 유사도"
GPT = "GPT-5"


def measure_similarity(sentence):
    start = perf_counter()
    white, black, white_match, black_match = score([sentence])[0]
    predictions = {f"유사도 {percent}%": classify(white, black, percent / 100, percent / 100)
                   for percent in THRESHOLDS}
    return {
        **predictions, "유사도 시간(초)": perf_counter() - start,
        "화이트 최고": white, "블랙 최고": black,
        "화이트 근접 표현": white_match, "블랙 근접 표현": black_match,
    }


def metrics(frame, method):
    pred = frame[method]
    truth = frame["정답"]
    counts = summarize([{"정답": row["정답"], "예측": row[method], "응답 시간(초)": row["GPT 시간(초)" if method == "GPT" else "유사도 시간(초)"]}
                        for row in frame.to_dict("records")])
    counts["미탐"] += int(((truth == "위험") & (pred == "판단 유보")).sum())
    counts["유보"] = int((pred == "판단 유보").sum())
    return counts


def average_similarity(frame):
    per_threshold = [metrics(frame, f"유사도 {percent}%") for percent in THRESHOLDS]
    return {field: mean(result[field] for result in per_threshold) for field in per_threshold[0]}


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
st.caption("유사도 정확도는 화이트·블랙 공통 임계값 20·40·60·80%에서 각각 채점한 뒤 네 값의 평균으로 표시해요.")
progress = st.progress(len(results) / len(examples), text=f"{len(results)}/{len(examples)}문장 검증 완료")
if len(results) < len(examples):
    count = min(BATCH_SIZE - len(results) % BATCH_SIZE, len(examples) - len(results))
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
                reference_vectors()
                for sample in examples[len(results):len(results) + count]:
                    result = measure(sample, key)
                    results.append({"원문": result["원문"], "정답": result["정답"], "GPT": result["예측"],
                                    "GPT 시간(초)": result["응답 시간(초)"], "응답 모델": result["모델"],
                                    "유형": sample["category"], "문장 쌍": sample["pair_id"],
                                    **measure_similarity(sample["sentence"])})
                    progress.progress(len(results) / len(examples), text=f"{len(results)}/{len(examples)}문장 검증 완료")
            except (HTTPError, OSError, ValueError, KeyError) as error:
                st.error(f"검증이 중단됐어요 ({getattr(error, 'code', type(error).__name__)}). 남은 문장부터 이어갈 수 있어요.")
            else:
                st.rerun()

st.subheader("정확도와 문장당 평균 처리 시간")
if not results:
    st.info("첫 검증을 실행하면 두 방식의 실측 결과가 표시돼요.")
    st.stop()
reference_vectors()
for row in results:
    if "유사도 시간(초)" not in row:
        row.update(measure_similarity(row["원문"]))
table = pd.DataFrame(results)
summary = pd.DataFrame([
    {"방식": SIMILARITY, **average_similarity(table)},
    {"방식": GPT, **metrics(table, "GPT")},
])
st.caption(f"두 방식이 모두 처리한 {len(table)}문장 기준 · 응답 모델 {results[-1]['응답 모델']}")
st.caption("유사도는 기준 벡터 준비가 끝난 뒤 문장 하나를 벡터화하고 네 임계값에서 판정하는 전체 시간을 재요. GPT-5는 API 왕복 시간을 재므로 네트워크 시간도 포함돼요.")
def completed_points(frame):
    points = []
    for end in range(BATCH_SIZE, len(frame) + 1, BATCH_SIZE):
        batch = frame.iloc[end - BATCH_SIZE:end]
        points.extend([
            {"완료 문장 수": end, "방식": SIMILARITY, **average_similarity(batch)},
            {"완료 문장 수": end, "방식": GPT, **metrics(batch, "GPT")},
        ])
    return points


points = completed_points(table)
st.caption("각 점은 해당 20문장의 결과예요. 예를 들어 X축 40의 점은 21~40번 문장을 집계한 값이에요.")
if points:
    chart_data = pd.DataFrame(points)
    a, b = st.columns(2)
    for column, metric, title, scale in [
        (a, "정확도", "정확도 (%)", alt.Scale(domain=[0, 100])),
        (b, "문장당 시간(초)", "문장당 평균 처리 시간 (초)", alt.Scale(zero=True)),
    ]:
        chart = alt.Chart(chart_data).mark_line(point=alt.OverlayMarkDef(filled=True, size=75)).encode(
            x=alt.X("완료 문장 수:Q", axis=alt.Axis(values=list(range(BATCH_SIZE, len(examples) + 1, BATCH_SIZE)), format="d"),
                    scale=alt.Scale(domain=[BATCH_SIZE, len(examples)], nice=False)),
            y=alt.Y(f"{metric}:Q", title=title, scale=scale),
            color=alt.Color("방식:N", scale=alt.Scale(domain=[SIMILARITY, GPT], range=["#235aa6", "#aa4d14"])),
            tooltip=["완료 문장 수:Q", "방식:N", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        column.altair_chart(chart.properties(title=title, height=230), use_container_width=True)
else:
    st.info("첫 20문장의 검증이 끝나면 그래프에 두 방식의 첫 점이 표시돼요.")
st.caption("전체 검증 완료 문장 기준 누적 집계")
st.dataframe(summary, hide_index=True)
with st.expander("임계값별 정확도 · 오탐 · 미탐 · 유보"):
    st.dataframe(pd.DataFrame([
        {"임계값": f"{percent}%", **metrics(table, f"유사도 {percent}%")} for percent in THRESHOLDS
    ]).drop(columns="문장당 시간(초)"), hide_index=True)
st.caption("유사도의 오탐·미탐·유보도 네 조건의 평균 건수예요. 판단 유보는 정확도에서 오답으로 집계해요. 위험 문장을 유보하면 미탐에도 포함해요.")
complete_batches = len(table) // BATCH_SIZE
if complete_batches:
    st.subheader("20문장 묶음별 상세 결과")
    batch_number = st.selectbox(
        "검증 묶음",
        range(1, complete_batches + 1),
        index=complete_batches - 1,
        format_func=lambda n: f"{n}차 · {(n - 1) * BATCH_SIZE + 1}~{n * BATCH_SIZE}번",
        key=f"pilot_detail_batch_{complete_batches}",
    )
    start = (batch_number - 1) * BATCH_SIZE
    batch = table.iloc[start:start + BATCH_SIZE].copy()
    batch.insert(0, "문장 번호", range(start + 1, start + len(batch) + 1))
    st.markdown("**유형별 결과**")
    st.dataframe(pd.DataFrame([
        {"유형": category, "검증 문장": len(group), "방식": label, **result}
        for category, group in batch.groupby("유형", sort=False)
        for label, result in [(SIMILARITY, average_similarity(group)), (GPT, metrics(group, "GPT"))]
    ]), hide_index=True)
    st.markdown("**문장별 판단과 오탐·미탐**")
    percent = st.selectbox("문장별 유사도 판정을 볼 임계값", THRESHOLDS, format_func=lambda value: f"{value}%")
    prediction_column = f"유사도 {percent}%"
    for label, prediction in [("유사도", prediction_column), ("GPT", "GPT")]:
        batch[f"{label} 오류"] = [
            "오탐" if actual == "안전" and predicted == "위험"
            else "미탐 · 판단 유보" if actual == "위험" and predicted == "판단 유보"
            else "미탐" if actual == "위험" and predicted == "안전"
            else "판단 유보" if predicted == "판단 유보"
            else "—"
            for actual, predicted in zip(batch["정답"], batch[prediction], strict=True)
        ]
    columns = ["문장 번호", "유형", "원문", "정답", prediction_column, "유사도 오류", "GPT", "GPT 오류"]
    different, same, errors = st.tabs(["다르게 판단", "같게 판단", "오탐·미탐·유보"])
    with different:
        st.dataframe(batch.loc[batch[prediction_column] != batch["GPT"], columns], hide_index=True)
    with same:
        st.dataframe(batch.loc[batch[prediction_column] == batch["GPT"], columns], hide_index=True)
    with errors:
        st.dataframe(batch.loc[(batch["정답"] != batch[prediction_column]) | (batch["정답"] != batch["GPT"]), columns], hide_index=True)
else:
    st.info("20문장 검증을 마치면 해당 묶음의 유형별 결과와 오탐·미탐 사례를 볼 수 있어요.")

st.download_button("전체 문장별 결과 CSV 저장", table.to_csv(index=False).encode("utf-8-sig"), "pilot-results.csv", mime="text/csv")
