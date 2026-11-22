"""Evaluate the recovered original corpus; compare historical aggregate results."""
import csv
import io
import json
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import streamlit as st
from model_study import STUDY_MODEL, PROMPT, measure, parse_csv, summarize

st.set_page_config(page_title="개인정보 동의서 2,000문장 비교", layout="wide")
st.title("동일한 2,000문장에서 세 방식 비교")
st.caption("유사도 기반·GPT-3는 기존 실험 결과, GPT-5는 복원한 원본 문장으로 새로 평가합니다.")
source = st.file_uploader("평가 데이터 CSV (sentence,label · 안전/위험 또는 0/1)", type=["csv"])
data = source.getvalue() if source else Path(__file__).with_name("original_samples_2000.csv").read_bytes()
try:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if not {"sentence", "label"}.issubset(reader.fieldnames or []):
        raise ValueError("CSV 열 이름은 sentence,label이어야 합니다.")
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=["sentence", "label"]); writer.writeheader()
    for row in reader:
        writer.writerow({"sentence": row["sentence"], "label": {"0": "안전", "1": "위험"}.get(row["label"].strip(), row["label"].strip())})
    examples = parse_csv(buf.getvalue().encode("utf-8-sig"))
except (ValueError, UnicodeError, KeyError, AttributeError) as exc:
    st.error(str(exc)); st.stop()
corpus_hash = sha256(json.dumps(examples, ensure_ascii=False).encode()).hexdigest()
protocol_hash = sha256((STUDY_MODEL + PROMPT).encode()).hexdigest()
default_hash = Path(__file__).with_name("original_corpus_hash.txt").read_text().strip()
is_original = corpus_hash == default_hash
if not is_original:
    st.warning("기본 원본과 다른 데이터입니다. 이 데이터에는 과거 두 방식의 수치를 함께 표시하지 않습니다.")
counts = pd.Series([r["label"] for r in examples]).value_counts()
st.write(f"총 {len(examples):,}문장 · 안전 {counts.get('안전', 0):,} · 위험 {counts.get('위험', 0):,}")
st.download_button("평가 데이터 CSV 저장", data, "evaluation-data.csv", "text/csv")
with st.expander("평가 조건"):
    st.write(f"요청 모델: {STUDY_MODEL}. 문장당 한 번 요청하며 정답은 API에 보내지 않습니다.")
    st.code(PROMPT)
    st.caption("동일한 문장을 사용하지만 과거 프롬프트·실행 환경까지 동일하게 재현한 실험은 아닙니다. 처리 시간에는 API 왕복 시간이 포함됩니다.")
if st.session_state.get("original_source") != (corpus_hash, protocol_hash):
    st.session_state.original_source = (corpus_hash, protocol_hash)
    st.session_state.original_results = []
results = st.session_state.original_results
resume = st.file_uploader("이전에 저장한 문장별 결과 CSV로 이어하기", type=["csv"], key="resume")
if resume and st.button("저장 결과 불러오기"):
    try:
        saved = list(csv.DictReader(io.StringIO(resume.getvalue().decode("utf-8-sig"))))
        if not saved or len(saved) > len(examples): raise ValueError("결과 수가 올바르지 않습니다.")
        restored = []
        for i, r in enumerate(saved):
            if r["데이터 해시"] != corpus_hash or r["평가 설정 해시"] != protocol_hash: raise ValueError("데이터 또는 평가 설정이 다릅니다.")
            if r["원문"] != examples[i]["sentence"] or r["정답"] != examples[i]["label"] or r["예측"] not in ("안전", "위험"): raise ValueError("문장 순서 또는 판정이 올바르지 않습니다.")
            seconds = float(r["응답 시간(초)"])
            if not 0 <= seconds < 86400: raise ValueError("처리 시간이 올바르지 않습니다.")
            r["응답 시간(초)"] = seconds; restored.append(r)
        st.session_state.original_results = restored; st.rerun()
    except (ValueError, KeyError, UnicodeError) as exc: st.error(f"결과를 불러올 수 없습니다: {exc}")
progress = st.progress(len(results)/len(examples), text=f"{len(results)}/{len(examples)}문장 완료")
size = st.selectbox("이번에 실행할 문장 수", [20, 100, 200], index=0)
st.caption("실행 시 API 이용료가 발생합니다. 문장당 한 번 요청하므로 200문장 실행에도 시간이 걸릴 수 있습니다. 실행 후 CSV를 저장하면 세션이 종료돼도 이어갈 수 있습니다.")
if len(results) < len(examples) and st.button(f"다음 {min(size, len(examples)-len(results))}문장 평가", type="primary"):
    try: key = st.secrets.get("OPENAI_API_KEY", "")
    except (FileNotFoundError, KeyError): key = ""
    if not key: st.error("Streamlit Secrets의 OPENAI_API_KEY를 설정해 주세요.")
    else:
        try:
            for sample in examples[len(results):len(results)+size]:
                result = measure(sample, key)
                result.update({"데이터 해시": corpus_hash, "평가 설정 해시": protocol_hash})
                results.append(result)
                progress.progress(len(results)/len(examples), text=f"{len(results)}/{len(examples)}문장 완료")
        except (HTTPError, OSError, ValueError, KeyError, TimeoutError):
            st.error("평가가 중단됐습니다. 완료된 결과를 저장하고 남은 문장부터 이어갈 수 있습니다.")
        else: st.rerun()
if results:
    table = pd.DataFrame(results)
    st.download_button("문장별 결과 CSV 저장 · 이어하기용", table.to_csv(index=False).encode("utf-8-sig"), "gpt5-original-results.csv", "text/csv")
    stats = summarize(results)
    st.metric("GPT-5 정확도" + (" · 중간 결과" if len(results)<len(examples) else " · 최종"), f"{stats['정확도']:.2f}%")
    st.write(f"오탐 {stats['오탐']}건 · 미탐 {stats['미탐']}건 · 문장당 API 왕복 시간 {stats['문장당 시간(초)']:.4f}초")
    if is_original and len(results) == len(examples):
        comparison = pd.DataFrame({"방식": ["유사도 기반", "GPT-3", "GPT-5"], "정확도 (%)": [68.78, 76.05, stats["정확도"]], "결과 구분": ["기존 실험", "기존 실험", "이번 실측"]})
        st.subheader("동일한 2,000문장 · 분류 정확도 비교")
        st.bar_chart(comparison, x="방식", y="정확도 (%)", sort=False)
        st.dataframe(comparison, hide_index=True)
        comparison["문장당 평균 처리 시간 (초)"] = [0.02, 59.0, stats["문장당 시간(초)"]]
        st.subheader("문장당 평균 처리 시간 비교")
        import altair as alt
        speed_chart = alt.Chart(comparison).mark_bar().encode(
            x=alt.X("방식:N", sort=["유사도 기반", "GPT-3", "GPT-5"], title=None),
            y=alt.Y("문장당 평균 처리 시간 (초):Q", scale=alt.Scale(type="log"), title="문장당 평균 처리 시간 (초 · 로그 축)"),
            color=alt.Color("방식:N", legend=None),
            tooltip=["방식:N", alt.Tooltip("문장당 평균 처리 시간 (초):Q", format=".5f"), "결과 구분:N"],
        )
        st.altair_chart(speed_chart, use_container_width=True)
        st.dataframe(comparison[["방식", "문장당 평균 처리 시간 (초)", "결과 구분"]], hide_index=True)
        st.caption("시간이 짧을수록 빠릅니다. 차이가 커 로그 축으로 표시했습니다. 유사도 0.02초·GPT-3 59초는 과거 실험 기록, GPT-5는 이번 문장별 API 왕복 시간의 평균입니다. 측정 시점·환경·호출 조건이 달라 모델 자체의 속도 향상률로 해석하지 않습니다.")
        st.download_button("세 방식 비교 CSV 저장", comparison.to_csv(index=False).encode("utf-8-sig"), "three-method-comparison.csv", "text/csv")
    elif is_original: st.info("2,000문장을 모두 평가하면 과거 두 방식과 비교하는 그래프가 표시됩니다.")
    points = [{"완료 문장 수": n, "누적 정확도 (%)": summarize(results[:n])["정확도"]} for n in range(20, len(results)+1, 20)]
    if points:
        st.subheader("GPT-5 누적 정확도")
        st.line_chart(pd.DataFrame(points), x="완료 문장 수", y="누적 정확도 (%)")
        st.subheader("GPT-5 누적 평균 처리 시간")
        time_points = [{"완료 문장 수": n, "누적 평균 처리 시간 (초)": summarize(results[:n])["문장당 시간(초)"]} for n in range(20, len(results)+1, 20)]
        st.line_chart(pd.DataFrame(time_points), x="완료 문장 수", y="누적 평균 처리 시간 (초)")
    with st.expander("문장별 결과"):
        st.dataframe(table.drop(columns=["데이터 해시", "평가 설정 해시"]), hide_index=True)
else: st.info("기본 원본 2,000문장이 준비됐습니다. 다음 문장 평가를 눌러 시작하세요.")
