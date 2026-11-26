"""Lehmus AI models: generator / judge LLMs and the embedding model."""

import hashlib
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

from . import config

_CALL_CACHE = config.CACHE / "lehmus_calls"
_CALL_CACHE.mkdir(parents=True, exist_ok=True)


def client() -> OpenAI:
    config.require("LEHMUS_URL", "LEHMUS_KEY")
    return OpenAI(base_url=config.LEHMUS_URL, api_key=config.LEHMUS_KEY, timeout=300, max_retries=3)


def async_client() -> AsyncOpenAI:
    config.require("LEHMUS_URL", "LEHMUS_KEY")
    return AsyncOpenAI(base_url=config.LEHMUS_URL, api_key=config.LEHMUS_KEY, timeout=300, max_retries=3)


def parse_json(text: str):
    """Parse the first JSON object in a model reply (tolerates ```json fences and chatter)."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def ask_json(prompt: str, model: str, system: str | None = None, max_tokens: int = 4000,
             temperature: float = 0.2) -> dict | None:
    """One cached chat call that should return a JSON object."""
    key = hashlib.sha256(json.dumps([model, system, prompt, temperature]).encode()).hexdigest()[:24]
    f: Path = _CALL_CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["parsed"]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    cl = client()
    for attempt in range(3):
        try:
            r = cl.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens,
                                           temperature=temperature)
            content = r.choices[0].message.content or ""
            parsed = parse_json(content)
            if parsed is not None:
                f.write_text(json.dumps({"raw": content, "parsed": parsed}, ensure_ascii=False),
                             encoding="utf-8")
                return parsed
        except Exception as e:  # network hiccup or server error: wait, then try again
            print(f"  Lehmus call failed ({type(e).__name__}: {e}); retrying")
            time.sleep(5 * (attempt + 1))
    return None


# ----------------------------------------------------------------- Ragas side
def ragas_llm(model: str, mode: str = "json", max_tokens: int = 8000, use_cache: bool = True):
    """Ragas judge LLM. use_cache=False for custom DiscreteMetrics, whose dynamic response model
    cannot be pickled by the disk cache."""
    import instructor
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory

    modes = {"json": instructor.Mode.JSON, "md_json": instructor.Mode.MD_JSON,
             "tools": instructor.Mode.TOOLS, "json_schema": instructor.Mode.JSON_SCHEMA}
    return llm_factory(model, provider="openai", client=async_client(), mode=modes[mode],
                       # one cache folder per judge model, so the second judge never reuses the first judge's answers
                       cache=DiskCacheBackend(cache_dir=str(config.CACHE / f"ragas_{model}")) if use_cache else None,
                       temperature=0, max_tokens=max_tokens)


def ragas_embeddings():
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import OpenAIEmbeddings

    return OpenAIEmbeddings(client=async_client(), model=config.MODEL_EMBED,
                            cache=DiskCacheBackend(cache_dir=str(config.CACHE / "ragas_emb")))
