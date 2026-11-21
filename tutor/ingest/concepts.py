"""TF-IDF key phrases used as concept tags on evidence units."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from ..text import STOPWORDS, tokenize

DISCOURSE = set("""welcome today first finally notice here simply lecture course demo summarise summary
pattern rather single number every iteration starts call called calls look see now next let lets way
usually really need means frequent small good recipe start values value whole new used make makes""".split())


def _analyzer(text: str) -> list[str]:
    toks = [t for t in tokenize(text) if not t[0].isdigit() and len(t) > 2 and t not in DISCOURSE]
    grams = list(toks)
    grams += [f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in STOPWORDS and b not in STOPWORDS]
    return grams


def tag_concepts(texts: list[str], top_k: int = 4, min_df: int = 1) -> list[list[str]]:
    if not texts:
        return []
    vec = TfidfVectorizer(analyzer=_analyzer, min_df=min_df, sublinear_tf=True)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return [[] for _ in texts]
    vocab = vec.get_feature_names_out()
    out = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        pairs = sorted(zip(row.indices, row.data), key=lambda p: -p[1])
        chosen: list[str] = []
        for idx, _ in pairs:
            term = vocab[idx]
            # prefer bigrams; skip unigrams already covered by a chosen bigram
            if any(term in c.split() for c in chosen if " " in c):
                continue
            chosen.append(term)
            if len(chosen) >= top_k:
                break
        out.append(chosen)
    return out
