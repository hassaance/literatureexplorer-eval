"""Step 2 (no endpoint calls): generate questions from the source documents, and check the hand-written ones.

Why custom generation instead of only the hand-written questions: hand-picked questions tend to be
about famous papers that any model (and any retriever) handles well. Here documents are sampled
*stratified by category from the document list* (so every category is covered, including the
obscure ones), and passages are taken at random from the body of each paper rather than from the
abstract. The endpoint is never consulted, so the test set cannot drift towards what it finds easy.

Generation uses one model and filtering uses another (default: GPT OSS 120B writes, Gemma4 checks),
so a question survives only if a second model agrees it is self-contained, non-trivial and supported.

    python 02_generate_testset.py --n-fact 40 --n-concept 12 --n-multi 10
    python 02_generate_testset.py --validate-handwritten

Outputs data/generated_candidates.jsonl (all candidates with keep/drop decisions) and
data/handwritten_validation.csv.
"""

import argparse
import json
import random
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from evalkit import config, corpus, lehmus

GEN_SYSTEM = ("You help build a test set for a question-answering assistant over research papers on "
              "AI engineering (LLMs, agents, RAG, evaluation). Always reply with a single JSON object.")

FACT_PROMPT = """Below is a passage from the paper "{title}".

Write ONE question that a software developer learning AI engineering could realistically ask, and
whose answer is stated in this passage.

Rules:
- The question must make sense without seeing the passage: name the paper, method or system it is about.
- It must NOT be answerable from the paper title alone. Prefer concrete details: design choices, steps,
  numbers, findings, limitations.
- Phrase it naturally; do not copy long phrases from the passage.
- The reference answer is 1-3 sentences and uses only information from the passage.
- If the passage is unusable (reference list, table fragment, garbled text), set "usable" to false.

Passage:
\"\"\"{passage}\"\"\"

Return JSON: {{"usable": true, "question": "...", "reference": "..."}}"""

CONCEPT_PROMPT = """Below is the beginning (title, abstract, introduction) of the paper "{title}".

Write ONE question a developer learning AI engineering might ask about the main problem, approach or
contribution of this paper, and a reference answer of 2-4 sentences based only on the text.
The question must name the paper or its method. Do not ask for the title or authors.

Text:
\"\"\"{passage}\"\"\"

Return JSON: {{"usable": true, "question": "...", "reference": "..."}}"""

MULTI_PROMPT = """Below are excerpts from two papers in the same area.

Paper A: "{title_a}"
\"\"\"{passage_a}\"\"\"

Paper B: "{title_b}"
\"\"\"{passage_b}\"\"\"

Write ONE question that needs information from BOTH papers to answer well (for example comparing
their goals, methods or findings). The question must name both papers or their methods. Give a
reference answer of 2-4 sentences using only the two excerpts. If the papers have nothing meaningful
in common, set "usable" to false.

Return JSON: {{"usable": true, "question": "...", "reference": "..."}}"""

CHECK_PROMPT = """You are checking a candidate test question for a QA system over research papers.

Source text:
\"\"\"{passage}\"\"\"

Question: {question}
Reference answer: {reference}

Answer each check strictly:
- self_contained: would a user understand which paper or method the question is about without the source text?
- supported: is every claim in the reference answer stated in the source text?
- trivial: can the question be answered from the paper title alone, or is it a yes/no question?
Return JSON: {{"self_contained": true/false, "supported": true/false, "trivial": true/false, "reason": "..."}}"""

VERIFY_PROMPT = """Check a hand-written reference answer against passages from the source paper "{title}".

Question: {question}
Reference answer: {reference}

Passages (retrieved by keyword overlap, may be incomplete):
\"\"\"{passages}\"\"\"

Is the reference answer supported by these passages? Use "partially" if some details are missing from
the passages or slightly wrong, and explain which ones. If anything is wrong, propose a corrected reference
that only uses the passages.
Return JSON: {{"verdict": "supported" | "partially" | "not_supported", "problems": "...", "corrected_reference": "..."}}"""


def stratified_docs(cov: pd.DataFrame, n: int, rng: random.Random, exclude=()) -> list:
    """Round-robin over categories so that every category is represented before any repeats."""
    by_cat = defaultdict(list)
    seen_titles, held_back = set(), []
    for _, r in cov.sample(frac=1, random_state=rng.randrange(10**6)).iterrows():
        if r.title.lower() in seen_titles:
            continue
        seen_titles.add(r.title.lower())
        (held_back if r.doc_id in exclude else by_cat[r.category]).append(r)
    cats = list(by_cat)
    rng.shuffle(cats)
    picked = []
    while len(picked) < n and any(by_cat.values()):
        for c in cats:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop())
    # only if there are not enough other documents, fall back to the excluded ones
    picked += held_back[: max(0, n - len(picked))]
    return picked


def check(cand: dict, passage: str, model: str) -> dict:
    v = lehmus.ask_json(CHECK_PROMPT.format(passage=passage[:7000], question=cand["question"],
                                            reference=cand["reference"]), model, max_tokens=3000, temperature=0)
    v = v or {}
    keep = bool(v.get("self_contained")) and bool(v.get("supported")) and not v.get("trivial", True)
    return {"check": v, "keep": keep}


def generate(args):
    rng = random.Random(args.seed)
    cov = pd.read_csv(config.DATA / "corpus_coverage.csv")
    usable = cov[cov.chars >= 3000].copy()
    print(f"documents with usable text: {len(usable)} of {len(cov)}")
    hw_docs = set()
    if config.HANDWRITTEN_JSONL.exists():
        for line in open(config.HANDWRITTEN_JSONL, encoding="utf-8"):
            hw_docs.update(json.loads(line)["source_doc_ids"])

    over = 1.4  # over-generate a bit because the filter drops some candidates
    jobs = []
    # Prefer documents not already covered by hand-written questions
    for r in stratified_docs(usable, int(args.n_fact * over), rng, exclude=hw_docs):
        jobs.append(("fact_generated", [r]))
    for r in stratified_docs(usable, int(args.n_concept * over), rng, exclude=hw_docs):
        jobs.append(("conceptual_generated", [r]))
    by_cat = defaultdict(list)
    for _, r in usable.drop_duplicates("title").iterrows():
        by_cat[r.category].append(r)
    pair_cats = [c for c, rs in by_cat.items() if len(rs) >= 2]
    for i in range(int(args.n_multi * over)):
        c = pair_cats[i % len(pair_cats)]
        a, b = rng.sample(by_cat[c], 2)
        jobs.append(("multi_doc_generated", [a, b]))

    out = []
    for qtype, docs in tqdm(jobs, desc="generating"):
        if qtype == "fact_generated":
            passage = corpus.random_body_passage(corpus.doc_text(docs[0].doc_id), rng)
            if not passage:
                continue
            prompt = FACT_PROMPT.format(title=docs[0].title, passage=passage)
        elif qtype == "conceptual_generated":
            passage = corpus.intro_passage(corpus.doc_text(docs[0].doc_id))
            prompt = CONCEPT_PROMPT.format(title=docs[0].title, passage=passage)
        else:
            pa = corpus.intro_passage(corpus.doc_text(docs[0].doc_id), 2500)
            pb = corpus.intro_passage(corpus.doc_text(docs[1].doc_id), 2500)
            passage = f"[{docs[0].title}] {pa}\n\n[{docs[1].title}] {pb}"
            prompt = MULTI_PROMPT.format(title_a=docs[0].title, passage_a=pa, title_b=docs[1].title, passage_b=pb)
        g = lehmus.ask_json(prompt, args.gen_model, system=GEN_SYSTEM, temperature=0.4)
        if not g or not g.get("usable", True) or not g.get("question") or not g.get("reference"):
            continue
        cand = {"question_type": qtype, "question": g["question"].strip(), "reference": g["reference"].strip(),
                "source_doc_ids": [d.doc_id for d in docs], "source_titles": [d.title for d in docs],
                "categories": [d.category for d in docs], "reference_context": passage,
                "expected_behavior": "answer", "origin": "generated",
                "generator": config.MODEL_NAMES.get(args.gen_model, args.gen_model)}
        cand.update(check(cand, passage, args.check_model))
        cand["checker"] = config.MODEL_NAMES.get(args.check_model, args.check_model)
        out.append(cand)

    # drop near-duplicate questions
    kept = []
    for c in out:
        if c["keep"] and any(fuzz.token_set_ratio(c["question"], k["question"]) > 85 for k in kept):
            c["keep"], c["check"]["reason"] = False, "near-duplicate of another question"
        if c["keep"]:
            kept.append(c)
    # trim to the requested numbers per type (in stratified order)
    target = {"fact_generated": args.n_fact, "conceptual_generated": args.n_concept,
              "multi_doc_generated": args.n_multi}
    count = defaultdict(int)
    for c in out:
        if c["keep"]:
            count[c["question_type"]] += 1
            if count[c["question_type"]] > target[c["question_type"]]:
                c["keep"], c["check"]["reason"] = False, "surplus (target count reached)"

    with open(config.GENERATED_JSONL, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    df = pd.DataFrame(out)
    print("\ncandidates per type (kept / total):")
    print(df.groupby("question_type")["keep"].agg(["sum", "count"]))
    print(f"categories covered by kept questions: "
          f"{df[df.keep].categories.explode().nunique()} of {cov.category.nunique()}")
    print(f"wrote {config.GENERATED_JSONL}")


def validate_handwritten(args):
    docs = pd.read_csv(config.DOCUMENTS_CSV).set_index("doc_id")
    rows = []
    items = [json.loads(line) for line in open(config.HANDWRITTEN_JSONL, encoding="utf-8")]
    for it in tqdm([i for i in items if i["needs_verification"] and i["source_doc_ids"]], desc="verifying"):
        text = "\n".join(corpus.doc_text(d) for d in dict.fromkeys(it["source_doc_ids"]))
        if len(text) < 2000:
            rows.append({"id": it["id"], "verdict": "no_source_text", "problems": "source file missing or empty"})
            continue
        passages = corpus.best_passages(text, it["question"] + " " + it["reference"], k=4)
        v = lehmus.ask_json(VERIFY_PROMPT.format(title=docs.loc[it["source_doc_ids"][0], "title"],
                                                 question=it["question"], reference=it["reference"],
                                                 passages="\n---\n".join(passages)),
                            args.verify_model, max_tokens=4000, temperature=0) or {}
        rows.append({"id": it["id"], "verdict": v.get("verdict", "judge_failed"),
                     "problems": v.get("problems", ""), "corrected_reference": v.get("corrected_reference", ""),
                     "question": it["question"], "reference": it["reference"]})
    out = pd.DataFrame(rows)
    out.to_csv(config.DATA / "handwritten_validation.csv", index=False)
    print(out["verdict"].value_counts())
    print("Review data/handwritten_validation.csv. Fix references in tools/make_handwritten.py where the "
          "verdict is not 'supported' (the passages are found by keyword search, so a 'partially' can be a "
          "false alarm), then run: python tools/make_handwritten.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fact", type=int, default=40)
    ap.add_argument("--n-concept", type=int, default=12)
    ap.add_argument("--n-multi", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gen-model", default=config.MODEL_LARGE)
    ap.add_argument("--check-model", default=config.MODEL_SMALL)
    ap.add_argument("--verify-model", default=config.MODEL_LARGE)
    ap.add_argument("--validate-handwritten", action="store_true",
                    help="only check the hand-written references against the PDFs")
    args = ap.parse_args()
    if args.validate_handwritten:
        validate_handwritten(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
