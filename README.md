# LiteratureExplorer evaluation (AI Engineering, Assignment 2)

This repo tests and evaluates the LiteratureExplorer RAG endpoint from the outside (black box). It has two parts:

1. **Output quality with Ragas**: faithfulness, answer relevancy, context precision and recall, and factual correctness, plus two custom Ragas metrics for questions the system should decline or correct.
2. **System tests**: coverage of the document collection, error handling and input validation, latency, consistency across repeated runs, and multi-turn behaviour.

All endpoint responses are cached and saved, so the results can be checked without calling the endpoint again.

## Setup

Requires Python 3.10 or newer. The Lehmus models are only reachable from the university network or over the **university VPN**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste the URLs and keys from the assignment page into .env
```

`.env` is in `.gitignore`. No URL or key is hard-coded in the scripts.

You also need the source files (`papers.zip`, about 1 GB) from the link on the assignment page. Put it in `data/papers.zip`. It is not committed.

## How to run

Run the steps in order. The endpoint request count for each step is listed so you stay under the limit of about 1,000. The client stops itself at `ENDPOINT_REQUEST_BUDGET` (default 950), never sends more than 2 requests at a time, and waits before retrying after a 429 or 502.

| Step | Command | Endpoint requests | What it does |
|---|---|---|---|
| 0 | `python 00_smoke_test.py` | 2 | Checks access and prints the response format. Check that chunks and their sources are parsed. |
| 1 | `python 01_prepare_corpus.py --zip data/papers.zip` | 0 | Extracts the text and matches each file to `data/documents.csv` (offline coverage check). |
| 2 | `python 02_generate_testset.py` | 0 | Generates questions from the PDFs (GPT OSS 120B) and filters them with a second model (Gemma4). |
| 2b | `python 02_generate_testset.py --validate-handwritten` | 0 | Checks the hand-written reference answers against the PDFs. |
| 3 | `python 03_build_testset.py` | 0 | Merges everything into `testset/testset.jsonl` and `.csv`. |
| 4 | `python 04_query_endpoint.py --limit 5`, then without `--limit` | about 2 × questions (about 240) | Sends each question to `/v1/chat/completions` and `/v1/context`. |
| 5 | `python 05_system_tests.py all` | about 270 | Runs coverage (about 180), errors (29), latency (20), determinism (30) and multi-turn (9) tests. |
| 6 | `python 06_ragas_eval.py --limit 3`, then `python 06_ragas_eval.py` | 0 | Runs the main Ragas evaluation, judged by GPT OSS 120B. |
| 6b | `python 06_ragas_eval.py --judge small --sample 30` | 0 | Runs a second judge (Gemma4) on a stratified subset to check judge agreement. |
| 6c | `python 06_ragas_eval.py --canary 15` | 0 | Runs the canary check: swapped answers and contexts must score low. |
| 7 | `python 07_analyze.py` | 0 | Writes `results/summary.md`, `results/per_question.csv` and `results/figures/`. |

Useful options:

- Every endpoint script accepts `--offline`. It rebuilds its outputs from the cache and makes no new requests.
- `05_system_tests.py latency --cold-wait-min 20` waits 20 minutes before measuring, to catch a cold start. `07_analyze.py` also flags any request made after more than 15 minutes of inactivity.
- `06_ragas_eval.py` writes each (question, metric) result as soon as it is done and skips finished pairs on a re-run. Failed jobs are retried on the next run. Judge calls are also cached in `.cache/`, with one folder per judge model. If most jobs fail with JSON or validation errors, add `--instructor-mode md_json`.
- Run `python 05_system_tests.py latency --cold-wait-min 20` again on another day to get a second cold-start sample.

## Test set design

The goal is a test set that is not tilted towards questions the system finds easy. The endpoint is never used to build the questions.

| Type | n | Source | Expected behaviour |
|---|---|---|---|
| `fact_generated` | ~40 | Random passage from the **body** of a paper (not the abstract, not the references). Documents are sampled round-robin over all 28 categories of the document list. | answer |
| `conceptual_generated` | ~12 | Abstract and introduction of a stratified sample of papers | answer |
| `multi_doc_generated` | ~10 | Two papers from the same category; the question needs both | answer |
| `fact_known_paper`, `conceptual_known_paper` | 24 | Hand-written, about well-known papers in the list; references checked against the PDFs (step 2b) | answer |
| `collection_level` | 5 | Hand-written, e.g. "which papers are about prompt injection?"; references taken from the document list | answer |
| `robustness_variant` | 10 | The same questions with typos, as keyword queries, paraphrased without the method name, or in Finnish | answer |
| `false_premise` | 6 | Hand-written, e.g. "Why did the Transformer use recurrent layers?" | correct the premise |
| `out_of_collection` | 6 | In-domain papers that are **not** in the collection (LoRA, the original RAG paper, ...), plus one invented paper | abstain |
| `off_topic`, `prompt_injection` | 6 | Off-topic requests, a system-prompt leak attempt, a phishing request | decline |

Generated questions go through a filter: a different model checks that each one is self-contained, non-trivial and supported by the passage, and near-duplicates are removed. Every candidate and the reason it was kept or dropped is in `data/generated_candidates.jsonl`. The final set, with the source document and reference passage for each question, is in `testset/testset.jsonl` and `testset/testset.csv`. The hand-written questions are defined in `tools/make_handwritten.py`, so they are easy to review.

## Why these Ragas metrics

The endpoint is a single-turn RAG system. Each answer comes with retrievable chunks, and the test set has reference answers. That decides which metrics apply.

| Metric | Used for | Why |
|---|---|---|
| Faithfulness | all answers | Hallucination check: are the answer's claims supported by the retrieved chunks? |
| Answer relevancy | answerable | Does the answer address the question? Uses the Lehmus embedding model. Questions are short, so its 512-token limit is not a problem. |
| Context precision (with reference) | answerable | Are the useful chunks ranked high? |
| Context recall | answerable, false premise | Do the chunks contain what the reference answer needs? Separates retrieval failures from generation failures. |
| Factual correctness (F1) | answerable | Claim-level agreement with the reference answer. |
| `behavior` (custom `DiscreteMetric`) | abstain / decline | Did it decline, answer from the context, answer beyond the context, or behave unsafely? |
| `premise` (custom `DiscreteMetric`) | false premise | Did it correct the premise, decline, or accept it? |

Not used, and why:

- Agent, tool-call, topic-adherence and multi-turn metrics: the endpoint has no tools, and multi-turn behaviour is tested separately.
- SQL, summarization and data-comparison metrics: wrong task.
- BLEU, ROUGE, exact match and string presence: free-text answers vary in wording.
- Context entity recall: entity overlap is a weak signal in scientific text.
- Noise sensitivity: it needs many judge calls per sample, and context precision and faithfulness already cover most of what it measures.

Next to Ragas, `07_analyze.py` computes judge-free retrieval metrics: hit@k and MRR of the question's source document among the `/v1/context` chunks. These need no LLM and are a useful check on the judge-based context metrics.

## How the judge is checked

1. **Second judge:** Gemma4 scores a stratified subset again. The summary reports Spearman correlation and mean absolute difference against GPT OSS 120B for each metric.
2. **Canaries:** for 15 questions, the answer (or the contexts) is swapped with those of an unrelated question. Faithfulness, factual correctness and context recall must drop sharply. If a metric still scores a canary high, it is not reliable.
3. **Keyword cross-check:** the judge's "declined" labels are compared with a simple keyword heuristic. Disagreements are listed for manual reading.
4. Generation and filtering of the test set use two different models, which reduces the risk of a model favouring its own outputs.

## Known limitations

- `/v1/context` is assumed to return the same chunks the chat route used. If the chat response does not include its sources, this cannot be verified from outside. `07_analyze.py` reports whether the chat route returned its own sources.
- LLM judges are noisy, and the CIs on small groups (n ≤ 10) are wide. The numbers support comparisons between question types, not small differences.
- Reference answers for generated questions come from one passage. Context recall can underrate a retrieval that found the same fact in another part of the paper.
- There was no human rating of the outputs (out of scope), no load testing (not allowed), and no usage logs to analyse (none exist yet).

## Files

```
00_smoke_test.py ... 07_analyze.py   pipeline steps (run in order)
evalkit/                             shared code: config, endpoint client (cache, budget, backoff), parsing, Lehmus, corpus
tools/make_handwritten.py            hand-written questions
data/documents.csv                   document list from XYZ Inc. (with doc_ids)
data/corpus_map.csv                  source file -> document mapping (step 1)
data/generated_candidates.jsonl      all generated questions, with keep/drop decisions
testset/testset.jsonl, .csv          final test set
raw/responses.jsonl                  endpoint answers and chunks for every test question
raw/system_*.jsonl                   raw results of the system tests
raw/http_cache/                      every endpoint response, as received
raw/request_log.jsonl                every network call to the endpoint (time, status, latency)
results/                             Ragas scores, summary.md, per_question.csv, figures
```
