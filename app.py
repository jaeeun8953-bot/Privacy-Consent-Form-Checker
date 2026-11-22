"""Streamlit entry point for the consent checker."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError, URLError

import streamlit as st

from run_review import request_review

PAGE = Path(__file__).with_name("index.html")
MAX_LENGTH = 30_000


def rule_engine():
    # Reuse the same sample and rules that power the GitHub Pages demo.
    html = PAGE.read_text(encoding="utf-8")
    return "const sample = " + html.split("const sample = ", 1)[1].split("const $ = id =>", 1)[0]


def run_rules(document):
    script = rule_engine() + """
const input = require('node:fs').readFileSync(0, 'utf8');
process.stdout.write(JSON.stringify(input ? analyze(input) : sample));
"""
    result = subprocess.run(
        ["node", "-e", script], input=document, encoding="utf-8",
        capture_output=True, check=True, timeout=10,
    )
    return json.loads(result.stdout)


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


def load_sample():
    st.session_state.document = run_rules("")


def render_rules(result):
    st.caption(f"검토 항목 {len(result['checks'])}건 · 고지 항목 점검 구역 {result['evaluated']}개")
    if not result["checks"]:
        st.info("설정된 규칙에서 추가 검토 항목을 찾지 못했어요.")
    report = ["개인정보 동의서 점검 · 기존 방식"]
    for check in result["checks"]:
        with st.container(border=True):
            st.markdown("**" + check["title"] + "**")
            st.text(check["detail"])
            quote = f"{check['line']['number']}행 · {check['line']['value']}"
            st.text(quote)
            st.caption("개인정보 보호법 " + check["law"])
            report.append("\n".join([check["title"], check["detail"], quote, check["law"]]))
    return "\n\n".join(report)


st.set_page_config(page_title="Privacy Consent Form Checker", layout="wide")
st.title("개인정보 동의서 점검")
st.caption("같은 동의서를 규칙 기반과 GPT 문맥 기반으로 비교해 보세요.")
left, right = st.columns(2, gap="large")

with left:
    st.button("샘플 불러오기", on_click=load_sample)
    document = st.text_area("동의서 원문", key="document", height=330, max_chars=MAX_LENGTH)
    st.caption("GPT 방식 실행 시 원문이 OpenAI API로 전송됩니다. 이름, 연락처 등 실제 개인을 식별할 수 있는 값은 지우고 실행하세요.")
    rules_column, gpt_column = st.columns(2)
    rules_clicked = rules_column.button("기존 방식 실행", type="primary")
    gpt_clicked = gpt_column.button("GPT 방식 실행")

fingerprint = sha256(document.encode("utf-8")).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state.fingerprint = fingerprint
    st.session_state.results = {}
results = st.session_state.results

with right:
    st.subheader("검토 결과")
    if rules_clicked or gpt_clicked:
        if not document.strip():
            st.info("점검할 동의서를 입력해 주세요.")
        elif rules_clicked:
            try:
                results["rules"] = run_rules(document)
            except (OSError, subprocess.SubprocessError, ValueError):
                st.error("규칙 엔진을 실행할 수 없습니다. 배포 환경의 Node.js 설치를 확인해 주세요.")
        else:
            try:
                key = st.secrets.get("OPENAI_API_KEY", "")
            except FileNotFoundError:
                key = ""
            if not key:
                st.info("앱 관리자가 Streamlit Secrets에 OpenAI API 키를 설정하면 GPT 방식을 사용할 수 있어요.")
            else:
                with st.spinner("GPT가 동의서를 검토하고 있어요…"):
                    try:
                        results["gpt"] = request_review(document, key)
                    except HTTPError as error:
                        st.error(api_error_message(error))
                    except (URLError, TimeoutError, ValueError):
                        st.error("GPT 검토를 완료하지 못했어요. 연결을 확인하거나 문서를 나눠 다시 시도해 주세요.")

    rules_tab, gpt_tab = st.tabs(["기존 방식", "GPT 방식"])
    with rules_tab:
        if "rules" in results:
            report = render_rules(results["rules"])
            st.download_button("기존 결과 TXT 저장", report, "consent-review-rules.txt")
        else:
            st.caption("기존 방식 실행을 누르면 결과가 표시됩니다.")
    with gpt_tab:
        if "gpt" in results:
            result = results["gpt"]
            st.caption(result["model"])
            st.text(result["review"])
            st.download_button("GPT 결과 TXT 저장", result["review"], "consent-review-gpt.txt")
        else:
            st.caption("GPT 방식 실행을 누르면 결과가 표시됩니다.")

st.caption("자동 검토 결과입니다. 원문과 실제 동의 화면을 함께 확인하세요.")
