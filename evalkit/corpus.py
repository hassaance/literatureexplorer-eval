"""Access to the extracted text of the source documents (after 01_prepare_corpus.py)."""

import random
import re
from functools import lru_cache

import pandas as pd

from . import config

REF_HEADING = re.compile(r"\n\s*(references|bibliography|reference list)\s*\n", re.I)
WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")


@lru_cache(maxsize=1)
def corpus_map() -> pd.DataFrame:
    if not config.CORPUS_MAP_CSV.exists():
        raise SystemExit("data/corpus_map.csv not found: run 01_prepare_corpus.py first.")
    return pd.read_csv(config.CORPUS_MAP_CSV)


@lru_cache(maxsize=512)
def doc_text(doc_id: str) -> str:
    cm = corpus_map()
    rows = cm[cm["doc_ids"].fillna("").str.split(";").map(lambda ids: doc_id in ids)]
    parts = []
    for tf in rows["text_file"]:
        p = config.CORPUS_TEXT_DIR / tf
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


def body_span(text: str) -> tuple[int, int]:
    """Start/end of the main body: skip the first ~8% (title page, abstract) and the references."""
    start = int(len(text) * 0.08)
    end = len(text)
    refs = list(REF_HEADING.finditer(text))
    if refs and refs[-1].start() > len(text) * 0.5:
        end = refs[-1].start()
    return start, end


def clean(s: str) -> str:
    s = re.sub(r"-\n(\w)", r"\1", s)  # undo hyphenation at line breaks
    return re.sub(r"\s+", " ", s).strip()


def passage_ok(p: str) -> bool:
    if len(p) < 600:
        return False
    alpha = sum(c.isalpha() for c in p) / len(p)
    cites = len(re.findall(r"et al\.|arXiv|\[\d+\]", p))
    return alpha > 0.68 and cites < 6


def windows(text: str, size: int = 1500, step: int = 750) -> list[str]:
    t = clean(text)
    return [t[i:i + size] for i in range(0, max(1, len(t) - size // 2), step)]


def random_body_passage(text: str, rng: random.Random, size: int = 1500, tries: int = 25) -> str | None:
    s, e = body_span(text)
    body = clean(text[s:e])
    if len(body) < size:
        return None
    for _ in range(tries):
        i = rng.randrange(0, len(body) - size)
        j = body.find(". ", i, i + 300)  # start at a sentence boundary
        i = j + 2 if j != -1 else i
        p = body[i:i + size]
        p = p[: p.rfind(". ") + 1] or p
        if passage_ok(p):
            return p
    return None


def intro_passage(text: str, size: int = 3500) -> str:
    return clean(text[:size * 2])[:size]


def best_passages(text: str, query: str, k: int = 3) -> list[str]:
    """Simple keyword-overlap ranking (good enough to find where a fact is stated)."""
    terms = set(WORD.findall(query.lower()))
    scored = []
    for w in windows(text):
        wl = w.lower()
        scored.append((sum(1 for t in terms if t in wl), w))
    scored.sort(key=lambda x: -x[0])
    return [w for _, w in scored[:k]]
