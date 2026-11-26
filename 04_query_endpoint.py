"""Step 4: send every test question to the endpoint (2 requests per question).

For each question: POST /v1/chat/completions (answer) and POST /v1/context (retrieved chunks).
Responses are cached, so re-running costs no new requests. Output: raw/responses.jsonl.

    python 04_query_endpoint.py --limit 5      # try a few first
    python 04_query_endpoint.py                # all questions
    python 04_query_endpoint.py --offline      # rebuild raw/responses.jsonl from cache only
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from evalkit import config
from evalkit.endpoint import EndpointClient, requests_used
from evalkit.parsing import answer_text, extract_chunks


def load_testset():
    return [json.loads(line) for line in open(config.TESTSET_JSONL, encoding="utf-8")]


def run_one(ep: EndpointClient, item: dict) -> dict:
    chat = ep.chat(item["question"])
    ctx = ep.context(item["question"])
    chat_extra = {k: v for k, v in (chat["json"] or {}).items() if k != "choices"} if isinstance(chat["json"], dict) else {}
    return {
        "id": item["id"],
        "question": item["question"],
        "answer": answer_text(chat["json"]),
        "chat_chunks": extract_chunks(chat_extra),       # sources the chat route itself reports, if any
        "context_chunks": extract_chunks(ctx["json"]),   # what /v1/context retrieves for the same query
        "chat": {k: chat.get(k) for k in ("status", "latency_s", "error", "requested_at", "from_cache", "attempts")},
        "context": {k: ctx.get(k) for k in ("status", "latency_s", "error", "requested_at", "from_cache", "attempts")},
        "chat_raw": chat.get("json") if chat.get("json") is not None else chat.get("text"),
        "context_raw": ctx.get("json") if ctx.get("json") is not None else ctx.get("text"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--offline", action="store_true", help="use cached responses only")
    args = ap.parse_args()
    if not args.offline:
        config.require("ENDPOINT_URL", "ENDPOINT_KEY")

    items = load_testset()
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.limit:
        items = items[: args.limit]
    ep = EndpointClient(offline=args.offline)
    before = requests_used()
    print(f"{len(items)} questions; endpoint requests used so far: {before}/{config.ENDPOINT_BUDGET}")

    with ThreadPoolExecutor(max_workers=config.ENDPOINT_MAX_PARALLEL) as pool:
        results = list(tqdm(pool.map(lambda it: run_one(ep, it), items), total=len(items)))

    out = config.RAW / "responses.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad = [r["id"] for r in results if r["chat"]["status"] != 200 or r["context"]["status"] != 200]
    print(f"wrote {out}. New endpoint requests: {requests_used() - before}. Non-200 responses: {len(bad)} {bad[:10]}")
    no_chunks = [r["id"] for r in results if r["context"]["status"] == 200 and not r["context_chunks"]]
    if no_chunks:
        print(f"WARNING: no chunks parsed for {len(no_chunks)} questions, e.g. {no_chunks[:5]}. "
              "Check the response format with 00_smoke_test.py.")


if __name__ == "__main__":
    main()
