"""Pull answers, retrieved chunks and their source documents out of endpoint responses.

The exact JSON shape of /v1/context is not documented, so the chunk extractor walks
the response and picks up any object that carries chunk text. Run
`python 00_smoke_test.py` once and check that it finds the chunks and sources.
"""

import re
import unicodedata
from functools import lru_cache
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

import pandas as pd
from rapidfuzz import fuzz

from . import config

TEXT_KEYS = ("text", "page_content", "chunk", "content", "passage", "snippet")
SOURCE_KEYS = ("x-amz-bedrock-kb-source-uri", "source_uri", "sourceUri", "uri", "s3Uri", "source",
               "url", "file", "filename", "file_name", "document", "document_id", "doc_id", "title",
               "name", "path", "key")
SCORE_KEYS = ("score", "relevance_score", "similarity", "distance")


def answer_text(chat_json) -> str:
    if not isinstance(chat_json, dict):
        return ""
    try:
        return chat_json["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        for k in ("answer", "output", "response", "text"):
            if isinstance(chat_json.get(k), str):
                return chat_json[k]
    return ""


def _text_of(d: dict):
    for k in TEXT_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):  # Bedrock style: {"content": {"text": ...}}
            inner = _text_of(v)
            if inner:
                return inner
    return None


def _find_first(d, keys, depth=0):
    """Depth-first search for the first non-empty string under one of `keys`."""
    if depth > 6:
        return None
    if isinstance(d, dict):
        for k in keys:
            v = d.get(k)
            if isinstance(v, (str, int, float)) and str(v).strip():
                return v
        for v in d.values():
            if isinstance(v, (dict, list)):
                found = _find_first(v, keys, depth + 1)
                if found is not None:
                    return found
    elif isinstance(d, list):
        for v in d:
            found = _find_first(v, keys, depth + 1)
            if found is not None:
                return found
    return None


def _strip_text_fields(d):
    if isinstance(d, dict):
        return {k: _strip_text_fields(v) for k, v in d.items() if k not in TEXT_KEYS}
    if isinstance(d, list):
        return [_strip_text_fields(v) for v in d]
    return d


def extract_chunks(js) -> list[dict]:
    """Return [{text, source, score}] for every retrieved chunk found in `js`."""
    out = []

    def walk(node):
        if isinstance(node, list):
            if node and all(isinstance(x, str) for x in node) and len(node) <= 50:
                for x in node:
                    if len(x) > 40:
                        out.append({"text": x, "source": None, "score": None})
                return
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            if "role" in node:  # a chat message, not a chunk
                return
            txt = _text_of(node)
            if txt and len(txt) > 40:
                meta = _strip_text_fields(node)
                src = _find_first(meta, SOURCE_KEYS)
                score = _find_first(meta, SCORE_KEYS)
                out.append({"text": txt, "source": str(src) if src is not None else None,
                            "score": float(score) if isinstance(score, (int, float)) else None})
                return
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)

    walk(js)
    # drop exact duplicates that appear twice in one response (e.g. citations + references)
    seen, uniq = set(), []
    for c in out:
        k = (c["text"][:200], c["source"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


# ------------------------------------------------------------- doc mapping
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def source_basename(source: str) -> str:
    if not source:
        return ""
    p = urlparse(source)
    path = unquote(p.path if p.scheme else source)
    return PurePosixPath(path).name.lower()


@lru_cache(maxsize=1)
def documents() -> pd.DataFrame:
    return pd.read_csv(config.DOCUMENTS_CSV)


@lru_cache(maxsize=1)
def corpus_map() -> pd.DataFrame | None:
    if config.CORPUS_MAP_CSV.exists():
        return pd.read_csv(config.CORPUS_MAP_CSV)
    return None


@lru_cache(maxsize=4096)
def source_to_docs(source: str) -> tuple:
    """Map a chunk's source (S3 URI, filename or title) to doc_ids from documents.csv."""
    if not source:
        return ()
    base = source_basename(source)
    cm = corpus_map()
    if cm is not None and base:
        hit = cm[cm["file_basename"].str.lower() == base]
        if len(hit):
            ids = ";".join(hit["doc_ids"].dropna().astype(str))
            return tuple(sorted({x for x in ids.split(";") if x}))
    # Fall back to fuzzy title match against the file name / title string
    target = norm(re.sub(r"\.(pdf|html?|txt|md)$", "", base or source))
    if len(target) < 8:
        return ()
    docs = documents()
    best, best_ids = 0, ()
    for title, grp in docs.groupby(docs["title"].map(norm)):
        sc = fuzz.token_set_ratio(target, title)
        if sc > best:
            best, best_ids = sc, tuple(grp["doc_id"])
    return best_ids if best >= 90 else ()


def chunk_doc_ids(chunks: list[dict]) -> list[tuple]:
    return [source_to_docs(c.get("source") or "") for c in chunks]


def first_hit_rank(chunks: list[dict], target_doc_ids) -> int | None:
    """1-based rank of the first chunk coming from one of `target_doc_ids`, else None."""
    targets = set(target_doc_ids)
    for i, ids in enumerate(chunk_doc_ids(chunks), start=1):
        if targets & set(ids):
            return i
    return None
