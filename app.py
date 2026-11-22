"""Streamlit entry point for the consent checker."""
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError, URLError

import streamlit as st

from run_review import INSTRUCTIONS, request_review

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


REVIEW_FIELDS = {
    "level": {"type": "string", "enum": ["위험", "주의"]},
    "title": {"type": "string"},
    "line": {"type": "integer"},
    "quote": {"type": "string"},
    "why": {"type": "string"},
    "fix": {"type": "string"},
}
REVIEW_FORMAT = {
    "type": "json_schema", "name": "consent_review", "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False, "required": ["summary", "findings"],
        "properties": {
            "summary": {"type": "string"},
            "findings": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": list(REVIEW_FIELDS), "properties": REVIEW_FIELDS,
            }},
        },
    },
}
REVIEW_INSTRUCTIONS = INSTRUCTIONS + """
출력은 제공된 JSON 스키마를 따른다. summary는 지금 가장 먼저 확인할 내용을 쉬운 한국어 한 문장으로 쓴다.
findings는 중요한 것부터 최대 6개만 남긴다. 위험은 동의 간주나 대리인 없이 아동 동의를 받는 등
문서에 드러난 우선 확인 사항에 쓴다. 주의는 빠지거나 모호한 항목, 추가 확인이 필요한 내용에 쓴다.
두 등급 모두 검토 우선순위로, 위법 여부를 확정하지 않는다. 문제가 보이지 않으면 findings를 비운다.
title은 짧게, why와 fix는 각각 한 문장으로 쓰고 fix는 바로 할 수 있는 수정 행동을 말한다.
quote는 해당 줄에서 그대로 옮긴 짧은 구절이어야 한다. 누락이거나 개인 식별값이 포함되어 있으면
quote를 비우고 관련 구역의 line을 쓴다. 해당 줄이 아예 없으면 line은 0으로 쓴다.
"""


def gpt_review(document, key):
    response = request_review(document, key, instructions=REVIEW_INSTRUCTIONS, response_format=REVIEW_FORMAT)
    result = json.loads(response["review"])
    lines = document.splitlines()
    for finding in result["findings"]:
        line = finding["line"]
        if not 1 <= line <= len(lines):
            finding["line"] = 0
            finding["quote"] = ""
        elif finding["quote"] not in lines[line - 1]:
            finding["quote"] = ""
    return result


def rules_review(document):
    result = run_rules(document)
    fixes = {
        "동의 간주 표현": "간주 문구를 빼고, 사용자가 직접 동의 여부를 선택할 수 있게 하세요.",
        "불명확하거나 장기인 보유 기간": "보유 기간이나 언제 파기하는지 구체적으로 적으세요.",
        "제공 대상이 포괄적으로 기재됨": "제공받는 기관의 이름을 적으세요.",
        "만 14세 미만 아동 동의 절차 확인": "동의가 필요한 경우 법정대리인의 동의를 받고 확인하는 절차를 안내하세요.",
        "수집·이용 고지 확인": "빠진 고지 항목을 수집·이용 동의 구역에 추가하세요.",
        "제3자 제공 고지 확인": "빠진 고지 항목을 제3자 제공 동의 구역에 추가하세요.",
    }
    findings = [{
        "level": "위험" if check["high"] else "주의",
        "title": check["title"],
        "line": check["line"]["number"],
        "quote": check["line"]["value"],
        "why": check["detail"],
        "fix": fixes.get(check["title"], ""),
        "law": check["law"],
    } for check in result["checks"]]
    if findings:
        summary = findings[0]["title"] + "부터 확인해 보세요."
    elif result["evaluated"]:
        summary = "설정된 규칙에서 추가 검토 항목을 찾지 못했어요."
    else:
        summary = "수집·이용 또는 제3자 제공 구역을 찾지 못했어요. 동의서 제목과 본문을 함께 붙여 넣어 주세요."
    return {"summary": summary, "findings": findings}


def render_review(result, label):
    findings = sorted(result["findings"], key=lambda item: item["level"] != "위험")
    st.markdown("<div class='review-summary'><b>한눈에 보기</b><br>" + escape(result["summary"]) + "</div>", unsafe_allow_html=True)
    risk = sum(item["level"] == "위험" for item in findings)
    st.caption(f"위험 {risk}건 · 주의 {len(findings) - risk}건")
    report = [label, result["summary"]]
    for item in findings:
        level = item["level"]
        tone = "risk" if level == "위험" else "caution"
        source = f"{item['line']}행" if item["line"] else ""
        if item["quote"]:
            source += " · " + item["quote"]
        evidence = ""
        if item["quote"]:
            evidence = f"<blockquote>{item['line']}행 · <mark>{escape(item['quote'])}</mark></blockquote>"
        elif item["line"]:
            evidence = f"<p class='review-law'>관련 구역 {item['line']}행</p>"
        fix = "<p><b>이렇게 고쳐요</b> " + escape(item["fix"]) + "</p>" if item["fix"] else ""
        law = "<p class='review-law'>개인정보 보호법 " + escape(item["law"]) + "</p>" if item.get("law") else ""
        st.markdown(
            f"<article class='review-card {tone}'><strong class='review-badge'>{level}</strong>"
            f"<b>{escape(item['title'])}</b>{evidence}"
            f"<p><b>왜 봐야 하나요</b> {escape(item['why'])}</p>{fix}{law}</article>",
            unsafe_allow_html=True,
        )
        report.append("\n".join(filter(None, [f"[{level}] {item['title']}", source, item["why"], item["fix"], item.get("law")])))
    return "\n\n".join(report)


st.set_page_config(page_title="Privacy Consent Form Checker", layout="wide")
st.markdown("""
<style>
.stApp {background:#f1f4f8;color:#182739}
.consent-header {background:#12263c;color:#fff;border-bottom:4px solid #44b9a8;
  border-radius:16px;padding:26px;margin-bottom:20px}
.consent-header h1 {color:#fff;margin:0;padding:0;font-size:clamp(26px,5vw,44px)}
.consent-header p {color:#d3e3ef;margin:10px 0 0}
.review-summary,.review-card {background:#fff;color:#182739;border:1px solid #dbe2eb;
  padding:14px;border-radius:10px;margin-bottom:12px;overflow-wrap:anywhere}
.review-card {border-left:4px solid #ab6314}
.review-card.risk {border-left-color:#b53e35}
.review-card p,.review-card blockquote {margin:8px 0 0}
.review-card blockquote {border:0;padding:8px;border-radius:6px;background:#f3f6fa;color:#182739}
.review-card mark {background:#ffe478;color:#182739}
.review-card.risk mark {background:#ffbcb4;color:#182739}
.review-badge {display:inline-block;margin-right:8px;background:#fff0cb;color:#784413;
  padding:1px 8px;border-radius:12px;font-size:.83rem}
.risk .review-badge {background:#ffe1dd;color:#8e251d}
.st-key-review-mode button[kind="primary"] {background:#fff;color:#173b6b;border:2px solid #173b6b}
.st-key-review-mode button[kind="secondary"] {background:#235aa6;color:#fff;border:2px solid #235aa6}
.st-key-review-mode button:focus-visible {outline:3px solid #44b9a8;outline-offset:2px}
.review-law {font-size:.85rem;color:#52647c}
</style>
<div class="consent-header"><h1>개인정보 동의서 점검</h1>
<p>빠진 고지 항목과 꼭 확인할 문구를 찾습니다.</p></div>
""", unsafe_allow_html=True)
st.caption("같은 동의서를 규칙 기반과 GPT 문맥 기반으로 비교해 보세요.")
left, right = st.columns(2, gap="medium")
left_panel = left.container(border=True)
right_panel = right.container(border=True)

with left_panel:
    st.button("샘플 불러오기", on_click=load_sample)
    document = st.text_area("동의서 원문", key="document", height=330, max_chars=MAX_LENGTH)
    st.caption("GPT 방식 실행 시 원문이 OpenAI API로 전송됩니다. 이름, 연락처 등 실제 개인을 식별할 수 있는 값은 지우고 실행하세요.")
    mode = st.session_state.setdefault("review_mode", "rules")
    with st.container(key="review-mode"):
        rules_column, gpt_column = st.columns(2)
        rules_clicked = rules_column.button(
            "✓ 기존 방식 실행" if mode == "rules" else "기존 방식 실행",
            key="execute-rules", type="primary" if mode == "rules" else "secondary",
            on_click=st.session_state.__setitem__, args=("review_mode", "rules"),
        )
        gpt_clicked = gpt_column.button(
            "✓ GPT 방식 실행" if mode == "gpt" else "GPT 방식 실행",
            key="execute-gpt", type="primary" if mode == "gpt" else "secondary",
            on_click=st.session_state.__setitem__, args=("review_mode", "gpt"),
        )

fingerprint = sha256(document.encode("utf-8")).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state.fingerprint = fingerprint
    st.session_state.results = {}
results = st.session_state.results

with right_panel:
    st.subheader("검토 결과")
    if rules_clicked or gpt_clicked:
        if not document.strip():
            st.info("점검할 동의서를 입력해 주세요.")
        elif rules_clicked:
            try:
                results["rules"] = rules_review(document)
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
                        results["gpt"] = gpt_review(document, key)
                    except HTTPError as error:
                        st.error(api_error_message(error))
                    except (URLError, TimeoutError, ValueError):
                        st.error("GPT 검토를 완료하지 못했어요. 연결을 확인하거나 문서를 나눠 다시 시도해 주세요.")

    other = "gpt" if mode == "rules" else "rules"
    if other in results:
        other_label = "GPT 방식" if other == "gpt" else "기존 방식"
        st.button(
            other_label + " 결과 보기", key="switch-result",
            on_click=st.session_state.__setitem__, args=("review_mode", other),
        )
    label = "기존 방식" if mode == "rules" else "GPT 방식"
    st.caption(label + " 결과")
    if mode in results:
        report = render_review(results[mode], label)
        st.download_button(label + " 결과 TXT 저장", report, f"consent-review-{mode}.txt")
    else:
        st.caption(label + " 실행을 누르면 결과가 표시됩니다.")

st.caption("위험은 먼저 확인할 항목, 주의는 추가 확인할 항목이에요. 실제 동의 화면과 원문을 함께 확인하세요.")
