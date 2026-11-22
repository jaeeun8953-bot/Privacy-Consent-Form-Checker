"""Compare consent sentences against reference eojeol."""
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer

WHITE = (
    "파기합니다", "삭제합니다", "종료일까지", "종료일부터", "거부권",
    "선택란", "선택사항", "동의받습니다", "확인합니다", "구분하여",
    "별도로", "한정합니다",
)

BLACK = (
    "간주합니다", "자동으로", "영구", "무기한", "제휴사", "관계사",
    "불특정", "일괄", "포함됩니다", "강제합니다", "차단합니다", "무관하게",
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
