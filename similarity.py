"""Compare consent sentences against reference phrases."""
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer

WHITE = (
    "목적 범위에서만 이용", "신청에 필요한 항목만 수집", "연락처만 수집",
    "수집 항목을 안내", "이용 목적을 명시", "고지한 목적에 한하여",
    "배송 완료 후 파기", "회원 탈퇴 시 삭제", "보유 기간을 명시",
    "보유 기간 만료 후 파기", "목적 달성 시 지체 없이 파기", "종료일로부터 30일간 보관",
    "동의를 거부할 수 있습니다", "동의 여부를 직접 선택", "거부 시 불이익을 안내",
    "선택 항목에 동의하지 않아도 이용 가능", "기본 서비스 이용에 불이익 없음",
    "광고 수신은 선택 항목", "홍보 수신은 별도로 선택", "가입 동의와 별도로 선택",
    "제공받는 자의 명칭을 안내", "제공하는 항목을 명시", "제공 목적을 미리 알림",
    "제공받는 자의 보유 기간을 고지", "제3자 제공은 별도 동의",
    "법정대리인의 동의를 받습니다", "법정대리인의 동의 여부를 확인",
    "일반 개인정보와 구분하여 동의", "민감정보는 별도의 동의를 받습니다",
    "고유식별정보는 구분하여 동의받습니다",
)

BLACK = (
    "악용할 위험이 있습니다", "제3자에게 유출가능합니다", "제 3자에게 이전합니다", "무조건 유출됩니다",
    "불특정 제3자에게 공유", "제공 대상을 특정하지 않음", "관계사에 일괄 제공",
    "어느 협력사에든 전달", "동의 없이 외부에 전달", "제3자에게 무단으로 제공",
    "사전 고지 없이 다른 목적으로 사용", "목적과 무관하게 이용", "관계사의 홍보에 임의로 사용",
    "개인정보를 영구 보관", "삭제 시점을 정하지 않음", "무기한 저장",
    "회사가 필요하다고 판단할 때까지", "필요 시까지 보유",
    "회신이 없으면 동의한 것으로 간주", "가입과 동시에 자동 동의", "체크하지 않아도 동의 처리",
    "광고 수신을 거부하면 기본 서비스 제한", "선택 동의를 필수로 요구", "거부하면 모든 기능 차단",
    "법정대리인 없이 직접 동의", "보호자 확인 절차 생략", "아동 본인 동의만으로 처리",
    "민감정보 동의를 일반 동의에 포함", "고유식별정보를 별도 동의 없이 수집", "필요 여부와 관계없이 모든 항목 수집",
)


@lru_cache(maxsize=1)
def reference_vectors():
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), strip_accents=None)
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
