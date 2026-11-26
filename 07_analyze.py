"""Step 7: turn raw responses, system tests and Ragas scores into tables, figures and a summary.

No endpoint or judge calls. Outputs:
  results/summary.md          all tables for the report (copy what you need)
  results/per_question.csv    one row per question: answer, retrieval hit, all scores
  results/figures/*.png       figures for the report appendix

    python 07_analyze.py
"""

import difflib
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from evalkit import config
from evalkit.parsing import chunk_doc_ids, first_hit_rank

FIG = config.RESULTS / "figures"
FIG.mkdir(exist_ok=True)
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0"
ABSTAIN_RE = re.compile(r"(not (contain|include|mention|provide|cover|have)|no (information|mention|details)|"
                        r"cannot (answer|find|provide)|can't (answer|find|provide)|unable to|"
                        r"outside (the|my) (scope|knowledge)|don't have|do not have|not available in)", re.I)
REFLIST_RE = re.compile(r"(et al\.|arXiv:|\(\d{4}\)|In Proceedings|doi\.org|\[\d+\])")
NUMERIC = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]

lines: list[str] = []


def md(s=""):
    lines.append(s)


def table(df: pd.DataFrame, floatfmt="{:.2f}"):
    if df is None or not len(df):
        md("_no data_\n")
        return
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    md("| " + " | ".join(map(str, d.columns)) + " |")
    md("|" + "---|" * len(d.columns))
    for _, r in d.iterrows():
        md("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r.values) + " |")
    md()


def readjsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and not pd.isna(v)], dtype=float)
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------- sections
def section_testset(tests):
    md("## 1. Test set\n")
    t = tests.groupby(["question_type", "expected_behavior"]).size().reset_index(name="n")
    table(t)
    cats = tests["categories"].explode().dropna()
    md(f"Questions with a source document cover **{cats.nunique()}** of "
       f"{pd.read_csv(config.DOCUMENTS_CSV).category.nunique()} categories in the document list; "
       f"**{tests['source_doc_ids'].explode().dropna().nunique()}** distinct documents.\n")


def section_retrieval(per_q):
    md("## 2. Retrieval (no judge needed)\n")
    md("Hit@k = the source document of the question appears among the k chunks returned by /v1/context. "
       "Only for questions with a known source document.\n")
    r = per_q[per_q["has_source"]]
    if not len(r):
        md("_no data_\n")
        return
    g = r.groupby("question_type").agg(n=("id", "size"), hit_any=("hit", "mean"),
                                      hit_at_1=("hit_at_1", "mean"), mrr=("rr", "mean"),
                                      all_sources_hit=("all_sources_hit", "mean"),
                                      chunks=("n_chunks", "median")).reset_index()
    tot = pd.DataFrame([{"question_type": "ALL", "n": len(r), "hit_any": r.hit.mean(), "hit_at_1": r.hit_at_1.mean(),
                         "mrr": r.rr.mean(), "all_sources_hit": r.all_sources_hit.mean(),
                         "chunks": r.n_chunks.median()}])
    table(pd.concat([g, tot]))
    md(f"Chunks per response (all questions): median {per_q.n_chunks.median():.0f}, "
       f"min {per_q.n_chunks.min():.0f}, max {per_q.n_chunks.max():.0f}. "
       f"Median chunk length: {per_q.chunk_chars_median.median():.0f} characters. "
       f"Share of retrieved chunks that look like reference lists: {per_q.reflist_share.mean():.0%}. "
       f"Responses containing duplicate chunks: {(per_q.dup_chunks > 0).mean():.0%}. "
       f"Chunks whose source could not be mapped to a listed document: {per_q.unmapped_share.mean():.0%}.\n")
    md(f"Chat route reports its own sources for {per_q.chat_has_chunks.mean():.0%} of questions "
       f"(if 0%, Ragas uses /v1/context chunks as a stand-in for what the generator saw).\n")


def section_ragas(per_q, scores):
    md("## 3. Ragas scores (judge: GPT OSS 120B)\n")
    if scores is None or not len(scores):
        md("_run 06_ragas_eval.py first_\n")
        return
    s = scores[scores.metric.isin(NUMERIC) & scores.error.isna()].copy()
    s["value"] = pd.to_numeric(s["value"], errors="coerce")
    rows = []
    for (qt, m), g in s.groupby(["question_type", "metric"]):
        lo, hi = boot_ci(g.value)
        rows.append({"question_type": qt, "metric": m, "n": g.value.notna().sum(), "mean": g.value.mean(),
                     "ci95": f"{lo:.2f}-{hi:.2f}" if not np.isnan(lo) else ""})
    for m, g in s[s.question_type.isin(per_q[per_q.behavior == "answer"].question_type.unique())].groupby("metric"):
        lo, hi = boot_ci(g.value)
        rows.append({"question_type": "ALL answerable", "metric": m, "n": g.value.notna().sum(),
                     "mean": g.value.mean(), "ci95": f"{lo:.2f}-{hi:.2f}"})
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index="question_type", columns="metric", values="mean").reindex(columns=NUMERIC)
    ns = long.groupby("question_type")["n"].max()
    wide.insert(0, "n", ns)
    table(wide.reset_index())
    md("<details><summary>with 95% bootstrap confidence intervals</summary>\n")
    table(long)
    md("</details>\n")
    errs = scores[scores.error.notna()]
    if len(errs):
        md(f"Metric jobs that failed (not included above): {len(errs)} — "
           f"{errs.error.str.split(':').str[0].value_counts().to_dict()}\n")
    # zero-score analysis: is a 0 a real failure or a metric artefact?
    f = s[s.metric == "faithfulness"]
    md(f"Faithfulness exactly 0: {(f.value == 0).sum()} answers; exactly 1: {(f.value == 1).sum()}. "
       f"Check the 0s in per_question.csv: a refusal ('the documents do not say...') has no claims and can "
       f"score 0 or NaN although it is the right behaviour.\n")

    md("### Decline / false-premise behaviour (custom Ragas DiscreteMetric)\n")
    b = scores[scores.metric.isin(["behavior", "premise"]) & scores.error.isna()]
    if len(b):
        table(b.groupby(["question_type", "metric", "value"]).size().reset_index(name="n"))
    # cross-check judge labels against a simple regex (cheap judge sanity check)
    bb = b[b.metric == "behavior"].merge(per_q[["id", "answer"]], on="id")
    if len(bb):
        bb["regex_declined"] = bb.answer.fillna("").map(lambda a: bool(ABSTAIN_RE.search(a)))
        agree = ((bb.value == "declined") == bb.regex_declined).mean()
        md(f"Judge label 'declined' vs. a keyword heuristic agree on {agree:.0%} of {len(bb)} answers "
           f"(disagreements are worth reading by hand).\n")

    # figure: one small panel per metric, question types as bars
    import matplotlib.pyplot as plt
    w = wide.drop(columns="n")
    w = w.dropna(how="all")
    if len(w):
        fig, axes = plt.subplots(1, len(NUMERIC), figsize=(13, 0.35 * len(w) + 1.4), sharey=True)
        for ax, m in zip(axes, NUMERIC):
            v = w[m]
            ax.barh(range(len(v)), v.fillna(0), color=BLUE, height=0.6)
            for i, x in enumerate(v):
                if not pd.isna(x):
                    ax.text(min(x, 0.85) + 0.02, i, f"{x:.2f}", va="center", fontsize=7, color=INK)
            ax.set_xlim(0, 1)
            ax.set_title(m.replace("_", " "), fontsize=9, color=INK)
            style(ax)
        axes[0].set_yticks(range(len(w)))
        axes[0].set_yticklabels([t.replace("_", " ") for t in w.index], fontsize=8)
        axes[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(FIG / "ragas_by_type.png", dpi=200)
        plt.close(fig)
        md("![Ragas by question type](figures/ragas_by_type.png)\n")


def section_judge(scores, small, canary):
    md("## 4. Can the judge be trusted?\n")
    if small is not None and len(small) and scores is not None:
        a = scores[scores.error.isna()][["id", "metric", "value"]]
        b = small[small.error.isna()][["id", "metric", "value"]]
        m = a.merge(b, on=["id", "metric"], suffixes=("_large", "_small"))
        rows = []
        for metric, g in m.groupby("metric"):
            if metric in NUMERIC:
                x = pd.to_numeric(g.value_large, errors="coerce")
                y = pd.to_numeric(g.value_small, errors="coerce")
                ok = x.notna() & y.notna()
                rho = stats.spearmanr(x[ok], y[ok]).statistic if ok.sum() > 2 else np.nan
                rows.append({"metric": metric, "n": int(ok.sum()), "spearman": rho,
                             "mean_abs_diff": (x[ok] - y[ok]).abs().mean(),
                             "mean_large": x[ok].mean(), "mean_small": y[ok].mean()})
            else:
                rows.append({"metric": metric, "n": len(g), "agreement": (g.value_large == g.value_small).mean()})
        md("Second judge (Gemma4 26B) on a stratified subset, compared with the main judge (GPT OSS 120B):\n")
        table(pd.DataFrame(rows))
    else:
        md("_run `06_ragas_eval.py --judge small --sample 30` for the judge-agreement check_\n")
    if canary is not None and len(canary):
        c = canary[canary.error.isna()].copy()
        c["value"] = pd.to_numeric(c.value, errors="coerce")
        c["base_id"] = c.id.str.split("#").str[0]
        c["kind"] = c.id.str.split("#").str[1]
        real = scores[scores.error.isna()].copy()
        real["value"] = pd.to_numeric(real.value, errors="coerce")
        rows = []
        for (kind, metric), g in c.groupby(["kind", "metric"]):
            r = real[(real.metric == metric) & real.id.isin(g.base_id)]
            rows.append({"canary": kind, "metric": metric, "n": len(g), "real_mean": r.value.mean(),
                         "canary_mean": g.value.mean(), "canary_scored_>=0.5": (g.value >= 0.5).mean()})
        md("Canaries (answer or contexts swapped with an unrelated question). A trustworthy metric should "
           "drop sharply:\n")
        table(pd.DataFrame(rows))
    else:
        md("_run `06_ragas_eval.py --canary 15` for the canary check_\n")


def section_robustness(per_q):
    md("## 5. Robustness to rephrasing\n")
    v = per_q[per_q.question_type == "robustness_variant"]
    if not len(v):
        md("_no data_\n")
        return
    base = per_q.set_index("id")
    rows = []
    for _, r in v.iterrows():
        b = base.loc[r.base_id] if r.base_id in base.index else None
        rows.append({"id": r.id, "variant": r.note, "question": r.question[:70],
                     "hit_rank": r.hit_rank, "base_hit_rank": None if b is None else b.hit_rank,
                     "factual_corr": r.get("factual_correctness"),
                     "base_factual_corr": None if b is None else b.get("factual_correctness")})
    table(pd.DataFrame(rows))


def section_latency(per_q):
    md("## 6. Latency and reliability\n")
    log = pd.DataFrame(readjsonl(config.RAW / "request_log.jsonl"))
    if not len(log):
        md("_no data_\n")
        return
    log["ts"] = pd.to_datetime(log.ts)
    log = log.sort_values("ts")
    log["gap_min"] = log.ts.diff().dt.total_seconds() / 60
    ok = log[log.status == 200]
    rows = []
    for route, g in ok.groupby("path"):
        x = g.latency_s
        rows.append({"route": route, "n": len(x), "median_s": x.median(), "p25_s": x.quantile(.25),
                     "p75_s": x.quantile(.75), "p90_s": x.quantile(.9), "p95_s": x.quantile(.95), "max_s": x.max()})
    md("All successful requests made during the evaluation (at most 2 in parallel):\n")
    table(pd.DataFrame(rows))
    seq = readjsonl(config.RAW / "system_latency.jsonl")
    if seq:
        s = pd.DataFrame(seq)
        s = s[s.status == 200]
        md("Sequential run (one request at a time):\n")
        table(s.groupby("route").latency_s.describe(percentiles=[.5, .9])[["count", "50%", "90%", "max"]].reset_index())
    cold = ok[ok.gap_min > 15]
    if len(cold):
        md(f"Requests after more than 15 min of inactivity (possible cold start): n={len(cold)}, "
           f"median {cold.latency_s.median():.1f} s vs. {ok[ok.gap_min <= 15].latency_s.median():.1f} s otherwise.\n")
    status = log.groupby("status", dropna=False).size().to_dict()
    md(f"Status codes of all {len(log)} network calls: {status}. Retries after 429/502: "
       f"{(log.attempt > 0).sum()}.\n")
    a = per_q.dropna(subset=["chat_latency"])
    if len(a) > 5:
        rho = stats.spearmanr(a.chat_latency, a.answer_chars).statistic
        md(f"Chat latency vs. answer length: Spearman rho = {rho:.2f} (n={len(a)}); median answer length "
           f"{a.answer_chars.median():.0f} characters.\n")

    import matplotlib.pyplot as plt
    routes = sorted(ok.path.unique())
    try:  # matplotlib >= 3.10
        orient = dict(orientation="horizontal")
        plt.boxplot([[0]], **orient)
        plt.close("all")
    except TypeError:
        orient = dict(vert=False)
    fig, ax = plt.subplots(figsize=(6, 1.0 + 0.6 * len(routes)))
    ax.boxplot([ok[ok.path == r].latency_s for r in routes], **orient, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#cde2fb", color=BLUE), medianprops=dict(color=BLUE, linewidth=2),
               whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
               flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED, markeredgecolor="none"))
    ax.set_yticks(range(1, len(routes) + 1))
    ax.set_yticklabels(routes, fontsize=8)
    ax.set_xlabel("latency (s)", fontsize=8, color=MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "latency.png", dpi=200)
    plt.close(fig)
    md("![Latency](figures/latency.png)\n")


def section_errors():
    md("## 7. Error handling and API behaviour\n")
    e = readjsonl(config.RAW / "system_errors.jsonl")
    if not e:
        md("_run 05_system_tests.py errors_\n")
        return
    d = pd.DataFrame(e)
    d["response"] = d.apply(lambda r: (r.answer or r.body_excerpt or "")[:110], axis=1)
    table(d[["case", "expected", "status", "pass", "leaks_internals", "latency_s", "response"]])
    md(f"Passed {d['pass'].sum()}/{len(d)}; server errors (5xx): {(d.status >= 500).sum()}; "
       f"error bodies exposing internals: {d.leaks_internals.sum()}.\n")


def section_coverage():
    md("## 8. Coverage of the document collection\n")
    cov_off = config.DATA / "corpus_coverage.csv"
    if cov_off.exists():
        c = pd.read_csv(cov_off)
        md(f"Offline (source files vs. list): {len(c)} list entries, {c.title.str.lower().nunique()} unique titles; "
           f"{(c.n_files == 0).sum()} entries without a source file; {c.low_text.sum()} with almost no "
           f"extractable text.\n")
        miss = c[(c.n_files == 0) | c.low_text]
        if len(miss):
            table(miss[["doc_id", "title", "category", "n_files", "chars"]])
    cm = config.CORPUS_MAP_CSV
    if cm.exists():
        m = pd.read_csv(cm)
        un = m[m.match_method == "unmatched"]
        if len(un):
            md(f"Files in the source ZIP that match no listed document: {len(un)}\n")
            table(un[["file", "chars"]])
    rows = readjsonl(config.RAW / "system_coverage.jsonl")
    if rows:
        d = pd.DataFrame(rows)
        d["hit"] = d.first_hit_rank.notna()
        md(f"Online title probe (/v1/context with the paper title): **{d.hit.sum()}/{len(d)}** documents "
           f"retrieved ({d.hit.mean():.0%}); top-1 for {(d.first_hit_rank == 1).sum()}.\n")
        g = d.groupby("category").hit.agg(["mean", "size"]).sort_values("mean")
        table(g.reset_index().rename(columns={"mean": "hit_rate", "size": "n_docs"}))
        md("Documents not found by their title:\n")
        table(d[~d.hit][["title", "category", "status", "n_chunks"]])
        all_src = pd.Series([s for r in rows for s in (r["sources"] or []) if s])
        md(f"Distinct source files seen across all title probes: {all_src.nunique()}.\n")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(g) + 1))
        ax.barh(range(len(g)), g["mean"], color=BLUE, height=0.6)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(g.index, g["size"])], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of documents retrieved by their own title", fontsize=8, color=MUTED)
        style(ax)
        fig.tight_layout()
        fig.savefig(FIG / "coverage_by_category.png", dpi=200)
        plt.close(fig)
        md("![Coverage](figures/coverage_by_category.png)\n")


def section_determinism():
    md("## 9. Consistency (same question, repeated)\n")
    rows = readjsonl(config.RAW / "system_determinism.jsonl")
    if not rows:
        md("_run 05_system_tests.py determinism_\n")
        return
    out = []
    for r in rows:
        ans = [a["answer"] for a in r["answers"] if a["answer"]]
        sims = [difflib.SequenceMatcher(None, a, b).ratio() for i, a in enumerate(ans) for b in ans[i + 1:]]
        ctx = r["context_runs"]
        out.append({"id": r["id"], "type": r["question_type"], "identical_answers": len(set(ans)) == 1,
                    "answer_similarity": np.mean(sims) if sims else np.nan,
                    "same_chunks": len(ctx) == 2 and ctx[0] == ctx[1]})
    d = pd.DataFrame(out)
    table(d)
    md(f"Identical answers in all 3 runs: {d.identical_answers.mean():.0%}; mean text similarity "
       f"{d.answer_similarity.mean():.2f}; identical retrieval: {d.same_chunks.mean():.0%}.\n")


def section_multiturn():
    md("## 10. Multi-turn behaviour\n")
    rows = readjsonl(config.RAW / "system_multiturn.jsonl")
    if not rows:
        md("_run 05_system_tests.py multiturn_\n")
        return
    for r in rows:
        md(f"- **{r['turn1']}** → **{r['turn2']}**  \n  with history: {r['answer2_with_history'][:300]!r}  \n"
           f"  without history: {r['answer2_without_history'][:300]!r}")
    md()


def section_examples(per_q, scores):
    md("## 11. Examples for the report (lowest scores)\n")
    if scores is None or not len(scores):
        return
    for metric in ("faithfulness", "factual_correctness", "context_recall"):
        if metric not in per_q:
            continue
        w = per_q[per_q[metric].notna()].nsmallest(3, metric)
        md(f"### lowest {metric}\n")
        for _, r in w.iterrows():
            reason = scores[(scores.id == r.id) & (scores.metric == metric)].reason.dropna()
            md(f"- **{r.id}** ({r.question_type}) {metric}={r[metric]:.2f}, hit rank={r.hit_rank}  \n"
               f"  Q: {r.question}  \n  A: {str(r.answer)[:400]!r}  \n  top sources: {r.top_sources}  \n"
               f"  judge: {str(reason.iloc[0])[:300] if len(reason) else ''}")
        md()


# ----------------------------------------------------------------------- main
def build_per_question(tests, responses, scores):
    rows = []
    resp = {r["id"]: r for r in responses}
    for _, t in tests.iterrows():
        r = resp.get(t.id)
        if r is None:
            continue
        chunks = r["context_chunks"]
        texts = [c["text"] for c in chunks]
        ids = chunk_doc_ids(chunks)
        src = t.source_doc_ids if isinstance(t.source_doc_ids, list) else []
        rank = first_hit_rank(chunks, src) if src else None
        # for multi-document questions: was every source paper retrieved?
        titles = pd.read_csv(config.DOCUMENTS_CSV).set_index("doc_id").title
        groups = {}
        for d in src:
            groups.setdefault(titles.get(d, d).lower(), set()).add(d)
        all_hit = all(any(g & set(x) for x in ids) for g in groups.values()) if groups else None
        rows.append({
            "id": t.id, "question_type": t.question_type, "behavior": t.expected_behavior, "question": t.question,
            "reference": t.reference, "answer": r["answer"], "note": t.note, "base_id": t.base_id,
            "has_source": bool(src) and t.expected_behavior in ("answer", "correct_premise"),
            "hit_rank": rank, "hit": rank is not None, "hit_at_1": rank == 1, "rr": 1 / rank if rank else 0.0,
            "all_sources_hit": float(all_hit) if all_hit is not None else np.nan, "n_chunks": len(chunks),
            "chunk_chars_median": np.median([len(x) for x in texts]) if texts else np.nan,
            "reflist_share": np.mean([len(REFLIST_RE.findall(x)) >= 4 for x in texts]) if texts else np.nan,
            "dup_chunks": len(texts) - len(set(x[:300] for x in texts)),
            "unmapped_share": np.mean([not x for x in ids]) if ids else np.nan,
            "top_sources": [c["source"] for c in chunks[:3]],
            "chat_has_chunks": bool(r["chat_chunks"]), "chat_status": r["chat"]["status"],
            "context_status": r["context"]["status"], "chat_latency": r["chat"]["latency_s"],
            "context_latency": r["context"]["latency_s"], "answer_chars": len(r["answer"] or ""),
            "regex_declined": bool(ABSTAIN_RE.search(r["answer"] or "")),
        })
    pq = pd.DataFrame(rows)
    if scores is not None and len(scores):
        s = scores[scores.error.isna()]
        wide = s.pivot_table(index="id", columns="metric", values="value", aggfunc="first")
        for c in wide.columns:
            if c in NUMERIC:
                wide[c] = pd.to_numeric(wide[c], errors="coerce")
        wide = wide.rename(columns={"behavior": "judge_behavior", "premise": "judge_premise"})
        pq = pq.merge(wide.reset_index(), on="id", how="left")
    return pq


def main():
    tests = pd.DataFrame(readjsonl(config.TESTSET_JSONL))
    responses = readjsonl(config.RAW / "responses.jsonl")
    if not responses:
        raise SystemExit("raw/responses.jsonl not found: run 04_query_endpoint.py first.")

    def load(name):
        rows = readjsonl(config.RESULTS / name)
        if not rows:
            return None
        d = pd.DataFrame(rows)
        # keep the latest successful result per (id, metric)
        d["ok"] = d.error.isna()
        d = d.sort_values("ok").drop_duplicates(["id", "metric"], keep="last").drop(columns="ok")
        return d

    scores, small, canary = load("ragas_large.jsonl"), load("ragas_small.jsonl"), load("ragas_large_canary.jsonl")
    per_q = build_per_question(tests, responses, scores)
    per_q.to_csv(config.RESULTS / "per_question.csv", index=False)

    md(f"# LiteratureExplorer evaluation summary\n\n_generated {datetime.now():%Y-%m-%d %H:%M}_\n")
    section_testset(tests)
    section_retrieval(per_q)
    section_ragas(per_q, scores)
    section_judge(scores, small, canary)
    section_robustness(per_q)
    section_latency(per_q)
    section_errors()
    section_coverage()
    section_determinism()
    section_multiturn()
    section_examples(per_q, scores)
    (config.RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {config.RESULTS / 'summary.md'}, per_question.csv and figures/")


if __name__ == "__main__":
    main()
