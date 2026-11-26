"""Step 5: black-box tests of the system (beyond answer quality).

    python 05_system_tests.py coverage      # ~180 req: can each listed document be retrieved by its title?
    python 05_system_tests.py errors        # ~30 req: invalid input, auth, unsupported parameters
    python 05_system_tests.py latency       # 20 req: sequential latency, no parallel load
    python 05_system_tests.py determinism   # 30 req: same question repeated
    python 05_system_tests.py multiturn     # 9 req: does the endpoint use conversation history?
    python 05_system_tests.py all
    python 05_system_tests.py latency --cold-wait-min 20   # wait first, to measure a cold start

All requests are cached; outputs go to raw/system_<test>.jsonl. Stress testing is deliberately
out of scope (assignment rule), so parallelism never exceeds ENDPOINT_MAX_PARALLEL.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from evalkit import config
from evalkit.endpoint import EndpointClient, requests_used
from evalkit.parsing import answer_text, extract_chunks, first_hit_rank, norm

CHAT, CTX = "/v1/chat/completions", "/v1/context"
LEAK_MARKERS = ("Traceback", "arn:aws", "botocore", "File \"/", "Exception", "stack", "lambda_function",
                "AccessDenied", "ValidationException", "bedrock")


def save(name: str, rows: list[dict]):
    out = config.RAW / f"system_{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {out} ({len(rows)} rows)")


def slim(rec: dict) -> dict:
    body = rec.get("json") if rec.get("json") is not None else rec.get("text")
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    return {"status": rec.get("status"), "latency_s": rec.get("latency_s"), "error": rec.get("error"),
            "requested_at": rec.get("requested_at"), "body_excerpt": (text or "")[:1500],
            "leaks_internals": any(m.lower() in (text or "").lower() for m in LEAK_MARKERS)
            and (rec.get("status") or 0) >= 400}


# --------------------------------------------------------------------- coverage
def test_coverage(ep, args):
    docs = pd.read_csv(config.DOCUMENTS_CSV)
    docs["t"] = docs["title"].map(norm)
    groups = docs.groupby("t", sort=False).agg(doc_ids=("doc_id", list), title=("title", "first"),
                                                 category=("category", "first")).reset_index(drop=True)
    if args.limit:
        groups = groups.head(args.limit)

    def run(r):
        rec = ep.context(r.title, tag="coverage")
        chunks = extract_chunks(rec.get("json"))
        return {"doc_ids": r.doc_ids, "title": r.title, "category": r.category, "n_chunks": len(chunks),
                "first_hit_rank": first_hit_rank(chunks, r.doc_ids),
                "sources": [c["source"] for c in chunks], **slim(rec)}

    with ThreadPoolExecutor(config.ENDPOINT_MAX_PARALLEL) as pool:
        rows = list(tqdm(pool.map(run, [r for r in groups.itertuples()]), total=len(groups), desc="coverage"))
    hit = sum(r["first_hit_rank"] is not None for r in rows)
    print(f"title probe: {hit}/{len(rows)} documents found among the retrieved chunks")
    save("coverage", rows)


# ----------------------------------------------------------------------- errors
def error_cases():
    q = "What is ReAct?"
    long_text = ("Retrieval-augmented generation combines a retriever with a generator. " * 450)  # ~31k chars
    very_long = long_text * 2  # ~62k chars, above Titan v2's 8k-token input limit
    return [
        # name, path, kwargs for EndpointClient.request, expected outcome, what it tells us
        ("no_auth_header", CHAT, dict(body={"messages": [{"role": "user", "content": q}]}, auth=False), "4xx", "auth enforced"),
        ("wrong_api_key", CHAT, dict(body={"messages": [{"role": "user", "content": q}]}, auth=False,
                                     headers={"Authorization": "Bearer wrong-key-123"}), "4xx", "auth enforced"),
        ("wrong_key_context", CTX, dict(body={"query": q}, auth=False, headers={"Authorization": "Bearer wrong"}), "4xx", "auth on both routes"),
        ("malformed_json", CHAT, dict(raw_body='{"messages": [{"role": "user", "content": "hi"}'), "4xx", "input validation"),
        ("missing_messages", CHAT, dict(body={"prompt": q}), "4xx", "input validation"),
        ("empty_messages", CHAT, dict(body={"messages": []}), "4xx", "input validation"),
        ("empty_content", CHAT, dict(body={"messages": [{"role": "user", "content": ""}]}), "4xx", "input validation"),
        ("only_system_message", CHAT, dict(body={"messages": [{"role": "system", "content": "You are helpful."}]}), "4xx", "input validation"),
        ("invalid_role", CHAT, dict(body={"messages": [{"role": "wizard", "content": q}]}), "4xx", "input validation"),
        ("content_not_string", CHAT, dict(body={"messages": [{"role": "user", "content": 12345}]}), "4xx", "input validation"),
        ("context_missing_query", CTX, dict(body={"text": q}), "4xx", "input validation"),
        ("context_empty_query", CTX, dict(body={"query": ""}), "4xx", "input validation"),
        ("context_query_number", CTX, dict(body={"query": 42}), "4xx", "input validation"),
        ("wrong_method_get", CHAT, dict(method="GET"), "4xx", "method handling"),
        ("unknown_route", "/v1/models", dict(method="GET"), "4xx", "OpenAI clients call /v1/models"),
        ("text_plain_content_type", CHAT, dict(body={"messages": [{"role": "user", "content": q}]},
                                               headers={"Content-Type": "text/plain"}), "any", "content-type handling"),
        ("long_input_31k_chat", CHAT, dict(body={"messages": [{"role": "user", "content": long_text + " " + q}]}), "2xx_or_4xx", "long input handled gracefully"),
        ("long_input_62k_context", CTX, dict(body={"query": very_long}), "2xx_or_4xx", "input above embedding limit"),
        # valid OpenAI-style variants a UI or SDK might send
        ("content_parts_format", CHAT, dict(body={"messages": [{"role": "user", "content": [{"type": "text", "text": q}]}]}), "2xx", "OpenAI content-parts format"),
        ("stream_true", CHAT, dict(body={"messages": [{"role": "user", "content": q}], "stream": True}), "2xx", "streaming support"),
        ("max_tokens_5", CHAT, dict(body={"messages": [{"role": "user", "content": q}], "max_tokens": 5}), "2xx", "is max_tokens respected?"),
        ("temperature_0", CHAT, dict(body={"messages": [{"role": "user", "content": q}], "temperature": 0}), "2xx", "parameter accepted"),
        ("model_field", CHAT, dict(body={"model": "gpt-4o", "messages": [{"role": "user", "content": q}]}), "2xx", "model field ignored or rejected?"),
        ("n_2", CHAT, dict(body={"messages": [{"role": "user", "content": q}], "n": 2}), "2xx", "n>1 handled?"),
        ("system_prompt_finnish", CHAT, dict(body={"messages": [{"role": "system", "content": "Always answer in Finnish."},
                                                                {"role": "user", "content": q}]}), "2xx", "is a client system prompt used?"),
        ("unicode_emoji", CHAT, dict(body={"messages": [{"role": "user", "content": "🤖 What is ReAct? Ünïcödé ✓"}]}), "2xx", "unicode"),
        ("html_script_tag", CHAT, dict(body={"messages": [{"role": "user", "content": "<script>alert(1)</script> What is ReAct?"}]}), "2xx", "is markup echoed back unescaped?"),
        ("sql_like_input", CTX, dict(body={"query": "'; DROP TABLE documents; -- What is ReAct?"}), "2xx", "odd characters in query"),
        ("single_char_query", CHAT, dict(body={"messages": [{"role": "user", "content": "?"}]}), "2xx", "meaningless input"),
    ]


def outcome_ok(expected: str, status) -> bool:
    if status is None:
        return False
    if expected == "4xx":
        return 400 <= status < 500
    if expected == "2xx":
        return 200 <= status < 300
    if expected == "2xx_or_4xx":
        return status < 500
    return True


def test_errors(ep, args):
    rows = []
    for name, path, kw, expected, purpose in tqdm(error_cases(), desc="errors"):
        kw = dict(kw)
        method = kw.pop("method", "POST")
        rec = ep.request(path, method=method, tag=f"err_{name}", **kw)
        s = slim(rec)
        rows.append({"case": name, "route": path, "method": method, "expected": expected, "purpose": purpose,
                     "pass": outcome_ok(expected, s["status"]), "answer": answer_text(rec.get("json")),
                     **s})
    df = pd.DataFrame(rows)
    print(df[["case", "expected", "status", "pass", "leaks_internals", "latency_s"]].to_string(index=False))
    save("errors", rows)


# ---------------------------------------------------------------------- latency
LAT_QUESTIONS = [
    "What is ReAct?", "How does Reflexion use verbal feedback?", "What is the LIMIT dataset?",
    "What benchmarks does WebArena include?", "What is HyDE?", "How does Voyager store skills?",
    "What is Agentic Context Engineering?", "What does AgentDojo evaluate?",
    "What failure modes do multi-agent systems have?", "How does G-Eval score summaries?",
]


def test_latency(ep, args):
    if args.cold_wait_min:
        print(f"waiting {args.cold_wait_min} min without requests to measure a cold start...")
        time.sleep(args.cold_wait_min * 60)
    tag = f"lat_{datetime.now():%Y%m%d%H%M}" if args.cold_wait_min else "lat"
    rows = []
    for i, q in enumerate(tqdm(LAT_QUESTIONS, desc="latency (sequential)")):
        for path in (CHAT, CTX):
            body = {"messages": [{"role": "user", "content": q}]} if path == CHAT else {"query": q}
            rec = ep.request(path, body, tag=tag)
            ans = answer_text(rec.get("json")) if path == CHAT else ""
            usage = (rec.get("json") or {}).get("usage") if isinstance(rec.get("json"), dict) else None
            rows.append({"route": path, "question": q, "seq": i, "cold_test": bool(args.cold_wait_min),
                         "answer_chars": len(ans), "usage": usage, **slim(rec)})
    df = pd.DataFrame(rows)
    print(df.groupby("route")["latency_s"].describe(percentiles=[.5, .9]).to_string())
    save("latency" + ("_cold" if args.cold_wait_min else ""), rows)


# ------------------------------------------------------------------ determinism
def test_determinism(ep, args):
    items = [json.loads(line) for line in open(config.TESTSET_JSONL, encoding="utf-8")]
    order = ["fact_known_paper", "fact_generated", "conceptual_generated", "conceptual_known_paper",
             "multi_doc_generated", "out_of_collection"]
    picked = []
    for t in order * 3:  # round-robin over types
        cand = [i for i in items if i["question_type"] == t and i not in picked]
        if cand and len(picked) < args.n_det:
            picked.append(cand[0])
    rows = []
    for it in tqdm(picked, desc="determinism"):
        runs = []
        for rep in range(3):  # rep 0 is the cached request from 04_query_endpoint.py
            tag = "" if rep == 0 else f"rep{rep}"
            rec = ep.chat(it["question"], tag=tag)
            runs.append({"rep": rep, "answer": answer_text(rec.get("json")), **slim(rec)})
        ctx = []
        for rep in range(2):
            rec = ep.context(it["question"], tag="" if rep == 0 else f"rep{rep}")
            ctx.append([c["text"][:200] for c in extract_chunks(rec.get("json"))])
        rows.append({"id": it["id"], "question_type": it["question_type"], "question": it["question"],
                     "answers": runs, "context_runs": ctx})
    save("determinism", rows)


# -------------------------------------------------------------------- multiturn
CONVERSATIONS = [
    ("What is ReAct?", "On which benchmarks was it evaluated?"),
    ("What does the Voyager paper propose?", "How does its skill library work?"),
    ("What is HyDE?", "Which encoder does it use to embed the generated document?"),
]


def test_multiturn(ep, args):
    rows = []
    for q1, q2 in tqdm(CONVERSATIONS, desc="multiturn"):
        r1 = ep.chat(q1, tag="mt")
        a1 = answer_text(r1.get("json"))
        hist = [{"role": "user", "content": q1}, {"role": "assistant", "content": a1}, {"role": "user", "content": q2}]
        r2 = ep.chat(messages=hist, tag="mt")
        r2_alone = ep.chat(q2, tag="mt_alone")
        rows.append({"turn1": q1, "answer1": a1, "turn2": q2,
                     "answer2_with_history": answer_text(r2.get("json")),
                     "answer2_without_history": answer_text(r2_alone.get("json")),
                     "status": [r1.get("status"), r2.get("status"), r2_alone.get("status")]})
    for r in rows:
        print(f"\nQ1: {r['turn1']}\nQ2: {r['turn2']}\n  with history:    {r['answer2_with_history'][:300]}\n"
              f"  without history: {r['answer2_without_history'][:300]}")
    save("multiturn", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["coverage", "errors", "latency", "determinism", "multiturn", "all"])
    ap.add_argument("--limit", type=int, help="coverage: only the first N documents")
    ap.add_argument("--n-det", type=int, default=10, help="determinism: number of questions")
    ap.add_argument("--cold-wait-min", type=float, default=0)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if not args.offline:
        config.require("ENDPOINT_URL", "ENDPOINT_KEY")
    ep = EndpointClient(offline=args.offline)
    before = requests_used()
    tests = ["coverage", "errors", "latency", "determinism", "multiturn"] if args.test == "all" else [args.test]
    for t in tests:
        globals()[f"test_{t}"](ep, args)
    print(f"\nnew endpoint requests: {requests_used() - before}; total used: {requests_used()}/{config.ENDPOINT_BUDGET}")


if __name__ == "__main__":
    main()
