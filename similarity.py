"""Sentence similarity and a small, separate demonstration benchmark."""
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer

WHITE = (
    "개인정보는 상담 예약과 본인 확인을 위해 수집합니다.",
    "수집하는 항목은 성명과 휴대전화 번호입니다.",
    "수집한 정보는 상담이 끝난 뒤 30일 동안 보관하고 파기합니다.",
    "정보주체는 개인정보 수집과 이용에 대한 동의를 거부할 수 있습니다.",
    "선택 정보 제공에 동의하지 않아도 기본 서비스를 이용할 수 있습니다.",
    "개인정보를 제공받는 자는 예시택배 주식회사입니다.",
    "제공받는 자는 상품 배송 목적으로 연락처를 이용합니다.",
    "제공받는 자는 배송 완료 후 30일간 보유합니다.",
    "만 14세 미만 아동은 법정대리인의 동의를 받고 이를 확인합니다.",
    "민감정보는 일반 개인정보 처리 동의와 구분하여 별도로 동의받습니다.",
    "고유식별정보는 다른 정보 처리에 대한 동의와 구분하여 별도로 동의받습니다.",
    "이벤트 홍보 수신 동의는 가입 동의와 별도로 선택할 수 있습니다.",
)

BLACK = (
    "회신이 없으면 개인정보 처리에 동의한 것으로 간주합니다.",
    "개인정보를 삭제하지 않고 영구 보관합니다.",
    "회사가 필요하다고 판단하는 동안 개인정보를 보유합니다.",
    "개인정보는 제휴사 및 관계사 등에 제공됩니다.",
    "만 14세 미만 아동은 법정대리인 없이 직접 동의할 수 있습니다.",
    "민감정보 처리 동의는 일반 개인정보 동의에 포함됩니다.",
    "고유식별정보는 별도의 동의 없이 일괄 수집합니다.",
    "광고 수신에 동의하지 않으면 기본 서비스를 이용할 수 없습니다.",
    "서비스 제공에 필요한 범위를 넘어 모든 정보를 수집합니다.",
    "회사는 이용자의 정보를 별도 고지 없이 다른 목적으로 사용합니다.",
    "동의에 체크하지 않아도 서비스 이용 시 자동으로 동의 처리됩니다.",
    "이용자의 정보를 불특정 제삼자에게 제공할 수 있습니다.",
)

@lru_cache(maxsize=1)
def reference_vectors():
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), strip_accents=None)
    return vectorizer, vectorizer.fit_transform(WHITE + BLACK)


def score(texts):
    if not texts:
        return []
    vectorizer, reference = reference_vectors()
    # TF-IDF rows are L2-normalized, so the dot product is cosine similarity.
    scores = (vectorizer.transform(texts) @ reference.T).toarray()
    results = []
    for row in scores:
        white = int(row[:len(WHITE)].argmax())
        black = int(row[len(WHITE):].argmax())
        results.append((float(row[white]), float(row[len(WHITE) + black]), WHITE[white], BLACK[black]))
    return results


def classify(white, black, white_threshold, black_threshold):
    """Screen against the whitelist, then assess the remaining candidates."""
    if white >= white_threshold:
        return "안전"
    if black >= black_threshold:
        return "위험"
    return "판단 유보"
