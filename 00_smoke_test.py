"""Step 0: check that the endpoint and the Lehmus models are reachable.

Uses 2 endpoint requests (both cached). Prints the response structure so you can
confirm that answers, chunks and chunk sources are parsed correctly.

    python 00_smoke_test.py            # endpoint + Lehmus
    python 00_smoke_test.py --skip-lehmus
"""

import argparse
import json

from evalkit import config
from evalkit.endpoint import EndpointClient, requests_used
from evalkit.parsing import answer_text, extract_chunks, source_to_docs


def shape(obj, depth=0, max_depth=4):
    """Short description of a JSON structure (keys and types, not the content)."""
    if depth >= max_depth:
        return "..."
    if isinstance(obj, dict):
        return {k: shape(v, depth + 1) for k, v in list(obj.items())[:15]}
    if isinstance(obj, list):
        return [shape(obj[0], depth + 1), f"... {len(obj)} items"] if obj else []
    if isinstance(obj, str):
        return f"str({len(obj)})"
    return type(obj).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lehmus", action="store_true")
    args = ap.parse_args()
    config.require("ENDPOINT_URL", "ENDPOINT_KEY")
    ep = EndpointClient()

    q = "What is WOLF?"
    print(f"== /v1/chat/completions  ({q})")
    r = ep.chat(q)
    print("status", r["status"], "latency", r["latency_s"], "cached", r["from_cache"])
    print(json.dumps(shape(r["json"]), indent=1))
    print("ANSWER:", answer_text(r["json"])[:600])
    ch = extract_chunks({k: v for k, v in (r["json"] or {}).items() if k != "choices"})
    print(f"chunks found in chat response (outside 'choices'): {len(ch)}")

    print(f"\n== /v1/context  ({q})")
    r = ep.context(q)
    print("status", r["status"], "latency", r["latency_s"], "cached", r["from_cache"])
    print(json.dumps(shape(r["json"]), indent=1))
    chunks = extract_chunks(r["json"])
    print(f"chunks parsed: {len(chunks)}")
    for c in chunks[:5]:
        print(f" - score={c['score']} source={c['source']} -> docs={source_to_docs(c['source'] or '')}")
        print(f"   {c['text'][:150]!r}")
    print(f"\nEndpoint requests used so far: {requests_used()} / budget {config.ENDPOINT_BUDGET}")

    if not args.skip_lehmus:
        from evalkit import lehmus
        cl = lehmus.client()
        for m in (config.MODEL_SMALL, config.MODEL_LARGE):
            out = cl.chat.completions.create(model=m, max_tokens=800,
                                             messages=[{"role": "user", "content": 'Reply with the JSON {"ok": true}'}])
            print(f"Lehmus {config.MODEL_NAMES[m]}: {out.choices[0].message.content!r}")
        emb = cl.embeddings.create(model=config.MODEL_EMBED, input="Bananas are yellow.")
        print(f"Lehmus embedding dim: {len(emb.data[0].embedding)}")


if __name__ == "__main__":
    main()
