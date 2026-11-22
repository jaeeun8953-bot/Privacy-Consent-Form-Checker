"""Streamlit dashboard for the consent classification experiment."""
from urllib.error import HTTPError

import altair as alt
import pandas as pd
import streamlit as st

from similarity import BLACK, WHITE
from validation import FILE_COUNT, METHODS, THRESHOLDS, sample_bytes, sample_name, validate


def api_error_message(error):
    try:
        code = json.loads(error.read(8192)).get("error", {}).get("code")
    except (ValueError, AttributeError):
        code = None
    if code == "insufficient_quota":
        return "OpenAI API 크레딧 또는 조직의 사용 한도를 확인해 주세요."
    if error.code == 429:
        return "OpenAI API 요청 한도에 도달했어요. 잠시 기다린 뒤 다시 시도해 주세요."
    if error.code in (401, 403):
        return "앱의 OpenAI API 키 또는 프로젝트의 모델 접근 권한을 확인해 주세요."
    return "OpenAI API에서 요청을 처리하지 못했어요. 잠시 후 다시 시도해 주세요."


st.set_page_config(page_title="개인정보 동의서 검증", layout="wide")
st.title("개인정보 동의서 위험 문장 검증")
st.caption("TF-IDF·코사인 유사도와 GPT에 같은 검증 파일을 넣어 비교해요. 파일 20개에 가상 문장 100개씩 있으며, 각각 안전 50개와 위험 50개예요.")

history = st.session_state.setdefault("research_100_phrases_history", [])
details = st.session_state.setdefault("research_100_phrases_details", {})
with st.expander("유사도 판정 기준"):
    st.markdown(
        "입력 문장과 각 기준 표현을 어절 내부의 2~4글자 조각을 쓰는 TF-IDF 벡터로 바꾼 뒤, 코사인 유사도로 비교해요. "
        "화이트리스트 최고 점수가 임계값에 도달하면 안전으로 분류해요. "
        "나머지는 후보로 넘겨 블랙리스트와 비교하고, 블랙리스트 임계값에 도달하면 위험, 그보다 낮으면 판단 유보로 분류해요."
    )
    st.caption("제3자 이전 등 일부 표현은 문맥을 추가로 확인할 신호로 사용해요.")

def reference_list(column, label, description, examples, key):
    with column:
        visible = st.session_state.get(key, False)
        action = "접기" if visible else "보기"
        st.button(
            f"{label} {action}", key=f"{key}_button",
            on_click=st.session_state.__setitem__, args=(key, not visible),
        )
        if visible:
            st.caption(f"{description} · 시연용 표현 {len(examples)}개")
            st.markdown("\n".join(f"{i}. {phrase}" for i, phrase in enumerate(examples, 1)))


white_column, black_column = st.columns(2)
reference_list(white_column, "화이트리스트", "안전 기준 표현", WHITE, "show_whitelist")
reference_list(black_column, "블랙리스트", "위험 기준 표현", BLACK, "show_blacklist")

st.markdown("**유사도 검증 임계값 · 20%, 40%, 60%, 80%**")
st.caption("각 회차에 화이트리스트와 블랙리스트에 같은 임계값을 적용해요. 파일별 유사도 점은 (20% 정확도 + 40% 정확도 + 60% 정확도 + 80% 정확도) ÷ 4예요. GPT는 파일당 한 번 평가해요.")

st.progress(len(history) / FILE_COUNT, text=f"검증 완료 {len(history)}/{FILE_COUNT}개")
if len(history) < FILE_COUNT:
    number = len(history) + 1
    a, b = st.columns(2)
    a.download_button("다음 검증 CSV 받기", sample_bytes(number), sample_name(number), mime="text/csv")
    if b.button(f"파일 {number:02} 검증", type="primary"):
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except FileNotFoundError:
            key = ""
        if not key:
            st.info("Streamlit Secrets에 OpenAI API 키를 설정해 주세요.")
        else:
            try:
                with st.spinner("두 방식으로 문장 100개를 검증하고 있어요…"):
                    result, rows = validate(number, key)
            except HTTPError as error:
                st.error(api_error_message(error))
            except (OSError, TimeoutError, ValueError, KeyError):
                st.error("검증을 완료하지 못했어요. 같은 파일을 다시 검증할 수 있어요.")
            else:
                history.append(result)
                details[number] = rows
                st.rerun()
    st.caption("검증을 누르면 가상 문장만 OpenAI API로 전달돼요. 정답 라벨은 앱에서 채점에 쓰이며, GPT 요청에는 비용이 발생할 수 있어요.")

st.subheader("파일별 정확도")
st.caption("정답과 일치한 문장 수 ÷ 100. 판단 유보는 정확도에서 오답으로 계산해요. 검증 결과는 현재 앱 세션에 유지돼요.")
if history:
    table = pd.DataFrame(history)
    chart_data = table.melt(id_vars=["파일 번호"], value_vars=METHODS, var_name="방식", value_name="정확도 (%)")
    chart = alt.Chart(chart_data).mark_line(point=alt.OverlayMarkDef(filled=True, size=80)).encode(
        x=alt.X("파일 번호:Q", scale=alt.Scale(domain=[1, FILE_COUNT]), axis=alt.Axis(values=list(range(1, FILE_COUNT + 1)))),
        y=alt.Y("정확도 (%):Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("방식:N", scale=alt.Scale(domain=METHODS, range=["#235aa6", "#aa4d14"])),
        tooltip=["파일 번호", "방식", alt.Tooltip("정확도 (%):Q", format=".2f")],
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(table[["파일 번호", *METHODS, *(f"{percent}% 정확도" for percent in THRESHOLDS)]], hide_index=True)
    with st.expander("파일별 오탐 · 미탐 · 판단 유보"):
        st.caption("오탐은 안전 문장을 위험으로 판정한 경우예요. 미탐은 위험 문장을 안전으로 판정하거나 판단을 유보한 경우예요.")
        st.dataframe(table, hide_index=True)
    file_number = st.selectbox("문장별 비교 파일", sorted(details), index=len(details) - 1)
    percent = st.selectbox("문장별 비교 임계값", THRESHOLDS, format_func=lambda value: f"{value}%")
    rows = details[file_number][percent]
    different, same, pending = st.tabs(["다르게 판단", "같게 판단", "판단 유보"])
    for tab, matches in [
        (different, [r for r in rows if r["유사도 판단"] != "판단 유보" and r["유사도 판단"] != r["GPT 판단"]]),
        (same, [r for r in rows if r["유사도 판단"] == r["GPT 판단"]]),
        (pending, [r for r in rows if r["유사도 판단"] == "판단 유보"]),
    ]:
        with tab:
            if matches:
                st.dataframe(matches, hide_index=True)
            else:
                st.caption("해당 문장이 없어요.")
    st.download_button("검증 결과 CSV 저장", table.to_csv(index=False).encode("utf-8-sig"), "accuracy-results.csv", mime="text/csv")
    if st.button("새로 검증 시작"):
        history.clear()
        details.clear()
        st.rerun()
else:
    st.info("파일 01을 검증하면 그래프에 두 방식의 첫 점이 표시돼요.")
