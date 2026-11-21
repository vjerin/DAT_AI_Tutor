"""Small text utilities shared by indexing, generation and verification."""
from __future__ import annotations

import hashlib
import re

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
this that these those it its it's into over under about above below than so such can could should would
will shall may might must do does did done have has had having not no nor only own same too very just
what which who whom whose when where why how all any both each few more most other some i me my we our
you your he him his she her they them their there here also one two use used using get got let lets
please tell explain show give mean means does doesn't dont don't im i'm s t re ve ll d
""".split())

# words that describe the question or the medium rather than the content asked about
QUESTION_META = set("""
slide slides lecture lectures video videos shown show shows according table code course instructor
mention mentioned explain explained describe described say says said talk talks part point time
called call use used typical page screen figure plot diagram displayed written visible lesson
happen happens happened work works much many kind type way thing things take takes taken
learner student students help helps helpful important purpose reason role benefit
benefits advantage advantages matter matters introduce introduced introduces discuss discussed cover covered
present presented
""".split())

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|\d+(?:\.\d+)?(?:e-?\d+)?")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?:e-?\d+)?")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lower-cased tokens that keep code identifiers intact.

    ``optimizer.zero_grad`` yields ``optimizer.zero_grad`` plus its parts
    ``optimizer``, ``zero_grad``, ``zero``, ``grad`` so both exact and partial
    matches work for code queries.
    """
    out: list[str] = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(tok)
        if tok[0].isdigit():
            continue
        parts = [p for p in tok.split(".") if p] if "." in tok else [tok]
        if len(parts) > 1:
            out.extend(parts)
        for p in parts:
            if "_" in p.strip("_"):
                out.extend(x for x in p.split("_") if x and x not in STOPWORDS)
    return out


def light_stem(tok: str) -> str:
    for suf in ("ization", "ations", "ation", "ings", "ing", "ness", "ies", "es", "ed", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def content_terms(text: str) -> set[str]:
    return {light_stem(t) for t in tokenize(text)}


def query_terms(text: str) -> list[str]:
    """Content terms of a question (stemmed), without question/medium words."""
    out = []
    for t in tokenize(text):
        if t in QUESTION_META:
            continue
        st = light_stem(t)
        if st not in out:
            out.append(st)
    return out


def numbers(text: str) -> set[str]:
    out = set()
    for n in NUMBER_RE.findall(text):
        try:
            out.add(repr(float(n)))
        except ValueError:
            pass
    return out


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def sha256_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def sha256_text(*chunks: str) -> str:
    return sha256_bytes(*(c.encode("utf-8") for c in chunks))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.\s]", " ", text.lower())).strip()


def ms_to_clock(ms: int | None) -> str:
    if ms is None:
        return ""
    s = int(ms // 1000)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
