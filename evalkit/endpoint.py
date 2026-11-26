"""Client for the LiteratureExplorer endpoint.

Every request goes through `EndpointClient.request`, which
  * caches responses on disk (same request + tag -> no second network call),
  * allows at most ENDPOINT_MAX_PARALLEL requests in flight,
  * backs off on 429/502/503/504 instead of retrying in a tight loop,
  * logs every real network call to raw/request_log.jsonl and refuses to go
    past the request budget (assignment limit: about 1,000 calls in total).
"""

import hashlib
import json
import random
import threading
import time
from datetime import datetime, timezone

import requests

from . import config

RETRY_STATUS = {429, 502, 503, 504}
BACKOFF_S = [5, 12, 25]

_lock = threading.Lock()
_slots = threading.BoundedSemaphore(config.ENDPOINT_MAX_PARALLEL)
REQUEST_LOG = config.RAW / "request_log.jsonl"
CACHE_DIR = config.RAW / "http_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class BudgetExceeded(RuntimeError):
    pass


def requests_used() -> int:
    if not REQUEST_LOG.exists():
        return 0
    with open(REQUEST_LOG, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EndpointClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 budget: int | None = None, offline: bool = False):
        self.base_url = (base_url or config.ENDPOINT_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else config.ENDPOINT_KEY
        self.budget = budget or config.ENDPOINT_BUDGET
        self.offline = offline  # True: only serve from cache, never call the network
        self.session = requests.Session()

    # ------------------------------------------------------------------ helpers
    def _key(self, method, path, body, headers, tag):
        blob = json.dumps([method, path, body, headers, tag], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _log_call(self, entry: dict) -> None:
        with _lock:
            with open(REQUEST_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _check_budget(self) -> None:
        used = requests_used()
        if used >= self.budget:
            raise BudgetExceeded(
                f"Request budget reached ({used}/{self.budget}). Raise ENDPOINT_REQUEST_BUDGET "
                "in .env only if you are sure you stay under the assignment limit.")

    # --------------------------------------------------------------------- main
    def request(self, path: str, body=None, *, method: str = "POST", raw_body: str | None = None,
                headers: dict | None = None, auth: bool = True, tag: str = "",
                use_cache: bool = True, timeout: float | None = None) -> dict:
        """Send one request. Returns a dict with status, latency_s, json, text, error.

        `body` is sent as JSON; `raw_body` is sent verbatim (for malformed-input tests).
        `tag` separates otherwise identical requests (e.g. repeat runs).
        """
        hdrs = {"Content-Type": "application/json"}
        if auth:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        if headers:
            hdrs.update(headers)
        # never store the real key in cache keys or logs
        safe_hdrs = {k: ("<key>" if v == f"Bearer {self.api_key}" else v) for k, v in hdrs.items()}
        key = self._key(method, path, body if raw_body is None else raw_body, safe_hdrs, tag)
        cache_file = CACHE_DIR / f"{key}.json"

        if use_cache and cache_file.exists():
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
            rec["from_cache"] = True
            return rec
        if self.offline:
            return {"status": None, "error": "not in cache (offline mode)", "json": None,
                    "text": "", "latency_s": None, "from_cache": False}

        url = self.base_url + path
        data = raw_body.encode() if raw_body is not None else (
            json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        attempts = []
        rec = None
        for attempt in range(len(BACKOFF_S) + 1):
            self._check_budget()
            with _slots:
                t0 = time.perf_counter()
                started = _now()
                try:
                    r = self.session.request(method, url, data=data, headers=hdrs,
                                             timeout=timeout or config.ENDPOINT_TIMEOUT_S)
                    latency = time.perf_counter() - t0
                    try:
                        js = r.json()
                    except ValueError:
                        js = None
                    rec = {"status": r.status_code, "latency_s": round(latency, 3), "json": js,
                           "text": r.text if js is None else "", "error": None,
                           "resp_headers": {k: v for k, v in r.headers.items()
                                            if k.lower() in ("content-type", "x-amzn-requestid",
                                                             "x-amzn-trace-id", "retry-after")}}
                except requests.RequestException as e:
                    latency = time.perf_counter() - t0
                    rec = {"status": None, "latency_s": round(latency, 3), "json": None,
                           "text": "", "error": f"{type(e).__name__}: {e}"}
            self._log_call({"ts": started, "method": method, "path": path, "tag": tag, "key": key,
                            "status": rec["status"], "latency_s": rec["latency_s"],
                            "attempt": attempt})
            attempts.append({"status": rec["status"], "latency_s": rec["latency_s"]})
            retryable = rec["status"] in RETRY_STATUS or (
                rec["status"] is None and "Timeout" not in (rec["error"] or ""))
            if not retryable or attempt == len(BACKOFF_S):
                break
            wait = BACKOFF_S[attempt] + random.uniform(0, 2)
            if rec.get("resp_headers", {}).get("Retry-After", "").isdigit():
                wait = max(wait, int(rec["resp_headers"]["Retry-After"]))
            time.sleep(wait)

        rec.update({"attempts": attempts, "requested_at": started, "path": path, "method": method,
                    "tag": tag, "request_body": body if raw_body is None else raw_body,
                    "from_cache": False})
        # Cache final outcomes, but not transient failures, so a later run can retry those.
        if use_cache and rec["status"] is not None and rec["status"] not in RETRY_STATUS:
            cache_file.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec

    # ------------------------------------------------------------ convenience
    def chat(self, question: str | None = None, messages: list | None = None, tag: str = "", **extra):
        body = {"messages": messages or [{"role": "user", "content": question}]}
        body.update(extra)
        return self.request("/v1/chat/completions", body, tag=tag)

    def context(self, query: str, tag: str = "", **extra):
        body = {"query": query}
        body.update(extra)
        return self.request("/v1/context", body, tag=tag)
