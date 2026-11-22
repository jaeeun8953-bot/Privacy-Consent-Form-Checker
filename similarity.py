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

# Separately authored examples with sentence-level labels. These are not legal determinations.
BENCHMARK = (
    ("예약 접수에 필요한 이름과 전화번호만 받습니다.", "안전"),
    ("회원 탈퇴가 완료되면 서비스 기록을 파기합니다.", "안전"),
    ("선택 항목에 동의하지 않아도 회원 가입이 가능합니다.", "안전"),
    ("수집한 정보는 이벤트 당첨 사실을 안내하는 데 사용합니다.", "안전"),
    ("배송 정보를 제공받는 곳은 가상물류 주식회사입니다.", "안전"),
    ("보존 기간은 상담 종료일부터 3개월입니다.", "안전"),
    ("신청자는 개인정보 수집 동의를 거부할 권리가 있습니다.", "안전"),
    ("법정대리인이 동의했는지 확인한 뒤 아동의 신청을 접수합니다.", "안전"),
    ("홍보 알림 수신 여부를 별도의 선택란에서 정할 수 있습니다.", "안전"),
    ("건강 관련 정보의 처리는 별도 동의란에서 선택합니다.", "안전"),
    ("가입 완료와 동시에 자동으로 동의한 것으로 봅니다.", "위험"),
    ("답변을 보내지 않은 회원의 정보 이용을 승인 처리합니다.", "위험"),
    ("등록한 정보는 삭제 시점을 정하지 않고 무기한 보유합니다.", "위험"),
    ("수집한 정보는 향후 회사 사정에 따라 필요한 만큼 보관합니다.", "위험"),
    ("연락처를 모든 협력 업체와 공유할 수 있습니다.", "위험"),
    ("14세 미만 이용자는 보호자 확인 절차 없이 신청할 수 있습니다.", "위험"),
    ("건강정보 처리를 일반 개인정보 수집 동의에 묶어 받습니다.", "위험"),
    ("여권번호 수집 동의는 회원 가입 동의 하나로 처리됩니다.", "위험"),
    ("광고 수신을 거절하면 회원의 기본 기능을 모두 차단합니다.", "위험"),
    ("다른 목적의 사용 여부는 이용자에게 알리지 않습니다.", "위험"),
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


def classify(white, black, threshold):
    if black >= threshold and black > white:
        return "위험"
    if white >= threshold and white > black:
        return "안전"
    return "판단 유보"


def metrics(examples, thresholds):
    scores = score([sentence for sentence, _ in examples])
    for threshold in thresholds:
        predictions = [classify(w, b, threshold) for w, b, *_ in scores]
        correct = sum(actual == prediction for (_, actual), prediction in zip(examples, predictions))
        yield {
            "임계값": threshold,
            "정확도 (%)": round(100 * correct / len(examples), 1),
            "판단 유보": predictions.count("판단 유보"),
            "오탐": sum(actual == "안전" and pred == "위험" for (_, actual), pred in zip(examples, predictions)),
            "미탐": sum(actual == "위험" and pred == "안전" for (_, actual), pred in zip(examples, predictions)),
            "위험 문장 유보": sum(actual == "위험" and pred == "판단 유보" for (_, actual), pred in zip(examples, predictions)),
        }
