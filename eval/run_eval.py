"""Retrieval evaluation: dense only vs hybrid (RRF) vs hybrid + reranker.

Usage:
  python -m eval.run_eval                                  # tuning set (30 questions)
  python -m eval.run_eval --file eval/questions_heldout.jsonl
  python -m eval.run_eval --file eval/questions_heldout.jsonl --failures
  python -m eval.run_eval --file eval/questions_heldout.jsonl --verify
"""
import argparse
import json
import time
from pathlib import Path

from src import hybrid
from src.hybrid import normalize_query

LANG = {"GDPR": "en", "UU PDP": "id"}
K = 5
POOL = 20


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def articles(chunks):
    """Ordered, de-duplicated article numbers from a ranked list of chunks."""
    seen = []
    for c in chunks:
        if c["article"] not in seen:
            seen.append(c["article"])
    return seen


def dense_only(q, lang):
    return [hybrid.BY_ID[i] for i in hybrid.dense_ids(q, POOL, lang)]


def hybrid_rrf(q, lang):
    ids = hybrid.rrf([hybrid.dense_ids(q, POOL, lang), hybrid.bm25_ids(q, POOL, lang)])
    return [hybrid.BY_ID[i] for i in ids]


def hybrid_rerank(q, lang):
    return [c for c, _ in hybrid.search(q, k=POOL, lang=lang, pool=POOL)]


CONFIGS = {
    "dense only": dense_only,
    "hybrid (RRF)": hybrid_rrf,
    "hybrid + reranker": hybrid_rerank,
}


def score(ranked_articles, gold):
    hit = any(a in gold for a in ranked_articles[:K])
    rr = 0.0
    for rank, a in enumerate(ranked_articles, 1):
        if a in gold:
            rr = 1.0 / rank
            break
    return hit, rr


def verify(questions):
    for item in questions:
        print(f"\nQ: {item['q']}  [{item['doc']}]")
        for g in item["gold"][:3]:
            for c in hybrid.CHUNKS:
                if c["doc"] == item["doc"] and c["article"] == g:
                    print(f"  gold {c['label']}: {c['text'][:90]}")
                    break


def evaluate(questions, show_failures):
    n = len(questions)
    rows, failures = [], {}
    first = questions[0]
    for name, fn in CONFIGS.items():
        fn(normalize_query(first["q"]), LANG[first["doc"]])  # warm-up, not timed
        hits, rrs, elapsed = 0, 0.0, 0.0
        failures[name] = []
        for item in questions:
            q = normalize_query(item["q"])
            t0 = time.perf_counter()
            chunks = fn(q, LANG[item["doc"]])
            elapsed += time.perf_counter() - t0
            ranked = articles(chunks)
            hit, rr = score(ranked, item["gold"])
            hits += hit
            rrs += rr
            if rr < 1.0:
                tag = "MISS" if not hit else "low "
                failures[name].append(
                    f"{tag} rr={rr:.2f} | gold={item['gold'][:3]} top5={ranked[:5]} | {item['q'][:70]}"
                )
        rows.append((name, hits / n, rrs / n, 1000 * elapsed / n))

    print(f"\nQuestions: {n}   K = {K}\n")
    print(f"{'config':<20} {'recall@' + str(K):<10} {'MRR':<8} {'ms/query':<8}")
    for name, rec, mrr, ms in rows:
        print(f"{name:<20} {rec:<10.3f} {mrr:<8.3f} {ms:<8.0f}")
    if show_failures:
        for name, lines in failures.items():
            print(f"\n== {name}")
            print("\n".join(lines) if lines else "(no failures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval/questions.jsonl")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--failures", action="store_true")
    args = ap.parse_args()
    questions = load(args.file)
    if args.verify:
        verify(questions)
    else:
        evaluate(questions, args.failures)


if __name__ == "__main__":
    main()
