"""Step 1 (offline, no endpoint calls): match the source files to the document list.

Reads papers.zip (or an unpacked folder), extracts the text of every file, and matches
each file to the entries in data/documents.csv. Outputs:

  data/corpus_map.csv            one row per file: which doc_id(s) it belongs to, how it matched
  data/corpus_coverage.csv       one row per listed document: is there a file? how much text?
  data/corpus_text/<file>.txt    extracted text (used to generate and check test questions)

    python 01_prepare_corpus.py --zip data/papers.zip
    python 01_prepare_corpus.py --dir path/to/unzipped/papers
"""

import argparse
import hashlib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from evalkit import config
from evalkit.parsing import norm

EXTS = (".pdf", ".html", ".htm", ".txt", ".md")
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.I)


def pdf_text(data: bytes) -> tuple[str, int]:
    try:
        import fitz  # PyMuPDF: faster and better layout handling, if installed
        fitz.TOOLS.mupdf_display_errors(False)  # malformed-PDF warnings are noise here
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc), doc.page_count
    except ImportError:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in r.pages), len(r.pages)


def html_text(data: bytes) -> str:
    html = data.decode("utf-8", errors="ignore")
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", html))


def iter_files(zip_path: Path | None, folder: Path | None):
    if zip_path:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                name = info.filename
                if info.is_dir() or "__MACOSX" in name or Path(name).name.startswith("."):
                    continue
                if name.lower().endswith(EXTS):
                    yield name, z.read(info)
    else:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                yield str(p.relative_to(folder)), p.read_bytes()


def match(fname: str, text: str, docs: pd.DataFrame):
    """Return (doc_ids, method, score) for the listed document this file belongs to."""
    head = text[:6000]
    base = Path(fname).stem
    # 1) arXiv id in the file name or first page
    for aid in ARXIV_RE.findall(base) + ARXIV_RE.findall(head[:3000]):
        hit = docs[docs["arxiv_id"] == aid]
        if len(hit):
            return hit, "arxiv_id", 100
    # 2) DOI
    for doi in DOI_RE.findall(head):
        doi = doi.rstrip(".,;)").lower()
        hit = docs[docs["doi"].fillna("").str.lower() == doi]
        if len(hit):
            return hit, "doi", 100
    # 3) fuzzy title match against file name and the start of the text
    nb, nh = norm(base), norm(head[:2500])
    best, best_title = 0, None
    for title in docs["title_norm"].unique():
        s = max(fuzz.token_set_ratio(title, nb) if len(nb) > 10 else 0,
                fuzz.partial_ratio(title, nh) if len(title) > 15 else 0)
        if s > best:
            best, best_title = s, title
    if best >= 88:
        return docs[docs["title_norm"] == best_title], "title_fuzzy", best
    return docs.iloc[0:0], "unmatched", best


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", type=Path)
    g.add_argument("--dir", type=Path)
    args = ap.parse_args()

    docs = pd.read_csv(config.DOCUMENTS_CSV)
    docs["title_norm"] = docs["title"].map(norm)
    docs["arxiv_id"] = docs["url"].str.extract(ARXIV_RE, expand=False)
    docs["doi"] = docs["url"].str.extract(DOI_RE, expand=False).str.rstrip("/ ")
    config.CORPUS_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, data in tqdm(list(iter_files(args.zip, args.dir)), desc="extracting"):
        pages = None
        try:
            if name.lower().endswith(".pdf"):
                text, pages = pdf_text(data)
            elif name.lower().endswith((".html", ".htm")):
                text = html_text(data)
            else:
                text = data.decode("utf-8", errors="ignore")
            err = ""
        except Exception as e:  # corrupt or encrypted file: that is a finding too
            text, err = "", f"{type(e).__name__}: {e}"
        hit, method, score = match(name, text, docs)
        # short name: Windows refuses paths longer than 260 characters
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem)[:50]
        txt_name = f"{stem}_{hashlib.md5(name.encode()).hexdigest()[:6]}.txt"
        (config.CORPUS_TEXT_DIR / txt_name).write_text(text, encoding="utf-8")
        alpha = sum(ch.isalpha() for ch in text[:20000]) / max(1, len(text[:20000]))
        rows.append({"file": name, "file_basename": Path(name).name.lower(), "text_file": txt_name,
                     "bytes": len(data), "pages": pages, "chars": len(text),
                     "alpha_ratio": round(alpha, 3), "doc_ids": ";".join(hit["doc_id"]),
                     "matched_title": hit["title"].iloc[0] if len(hit) else "",
                     "match_method": method, "match_score": score, "extract_error": err})

    cm = pd.DataFrame(rows)
    cm.to_csv(config.CORPUS_MAP_CSV, index=False)

    # Per listed document: is it in the source files at all, and with usable text?
    cov = []
    for _, d in docs.iterrows():
        files = cm[cm["doc_ids"].fillna("").str.split(";").map(lambda ids: d["doc_id"] in ids)]
        cov.append({"doc_id": d["doc_id"], "num": d["num"], "title": d["title"], "category": d["category"],
                    "url": d["url"], "n_files": len(files), "files": "; ".join(files["file"]),
                    "chars": int(files["chars"].sum()) if len(files) else 0,
                    "low_text": bool(len(files) and files["chars"].max() < 5000)})
    cov = pd.DataFrame(cov)
    cov.to_csv(config.DATA / "corpus_coverage.csv", index=False)

    dup_titles = docs[docs.duplicated("title_norm", keep=False)].groupby("title_norm")["doc_id"].apply(list)
    print(f"\nfiles in source: {len(cm)}   matched: {(cm.match_method != 'unmatched').sum()}   "
          f"unmatched: {(cm.match_method == 'unmatched').sum()}")
    print(f"listed documents: {len(docs)} ({docs.title_norm.nunique()} unique titles)")
    print(f"listed documents without a source file: {(cov.n_files == 0).sum()}")
    print(f"documents with very little text (<5000 chars, e.g. scanned or a web page stub): {cov.low_text.sum()}")
    print(f"documents listed twice: {len(dup_titles)} -> {dict(dup_titles)}")
    multi = cm[cm.doc_ids.fillna('').str.len() > 0].groupby("doc_ids").size()
    print(f"documents with more than one file: {(multi > 1).sum()}")
    print(f"\nwrote {config.CORPUS_MAP_CSV} and {config.DATA / 'corpus_coverage.csv'}")
    print("Check the 'unmatched' rows in corpus_map.csv by hand and fill in doc_ids if needed.")


if __name__ == "__main__":
    main()
