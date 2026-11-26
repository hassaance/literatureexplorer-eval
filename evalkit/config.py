"""Paths and settings. Secrets come from .env (never hard-coded)."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
TESTSET = ROOT / "testset"
RAW = ROOT / "raw"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"
for p in (DATA, TESTSET, RAW, RESULTS, CACHE):
    p.mkdir(exist_ok=True)

DOCUMENTS_CSV = DATA / "documents.csv"
CORPUS_MAP_CSV = DATA / "corpus_map.csv"
CORPUS_TEXT_DIR = DATA / "corpus_text"
TESTSET_JSONL = TESTSET / "testset.jsonl"
HANDWRITTEN_JSONL = TESTSET / "handwritten.jsonl"
GENERATED_JSONL = DATA / "generated_candidates.jsonl"

# Endpoint (system under test)
ENDPOINT_URL = os.getenv("ENDPOINT_BASE_URL", "").rstrip("/")
ENDPOINT_KEY = os.getenv("ENDPOINT_API_KEY", "")
ENDPOINT_BUDGET = int(os.getenv("ENDPOINT_REQUEST_BUDGET", "950"))
ENDPOINT_MAX_PARALLEL = int(os.getenv("ENDPOINT_MAX_PARALLEL", "2"))
ENDPOINT_TIMEOUT_S = float(os.getenv("ENDPOINT_TIMEOUT_S", "120"))

# Lehmus AI (judge, generator, embeddings)
LEHMUS_URL = os.getenv("LEHMUS_BASE_URL", "").rstrip("/")
LEHMUS_KEY = os.getenv("LEHMUS_API_KEY", "")
MODEL_SMALL = os.getenv("LEHMUS_MODEL_SMALL", "wnofibsjlomanbprgbdp")  # Gemma4 26B MoE
MODEL_LARGE = os.getenv("LEHMUS_MODEL_LARGE", "azsydsttjnlbfjbgqnwd")  # GPT OSS 120B
MODEL_EMBED = os.getenv("LEHMUS_MODEL_EMBED", "sxnzyzvwqxurdogkfxoe")
JUDGE_MAX_WORKERS = int(os.getenv("JUDGE_MAX_WORKERS", "4"))

MODEL_NAMES = {MODEL_SMALL: "Gemma4 26B MoE", MODEL_LARGE: "GPT OSS 120B"}


def require(*names: str) -> None:
    """Exit with a clear message if a required setting is missing."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        sys.exit(f"Missing settings: {', '.join(missing)}. Copy .env.example to .env and fill it in.")
