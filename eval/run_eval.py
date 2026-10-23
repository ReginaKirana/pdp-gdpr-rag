"""Retrieval evaluation: dense only vs hybrid (RRF) vs hybrid + reranker.
Usage:
  python -m eval.run_eval            # run the evaluation
  python -m eval.run_eval --verify   # print the gold chunks so you can check labels
"""
import json
import sys
from pathlib import Path
from src.hybrid import normalize_query

from src import hybrid

QUESTIONS = [json.loads(l) for l in Path("eval/questions.jsonl").open(encoding="utf-8")]
LANG = {"GDPR": "en", "UU PDP": "id"}
K = 5
POOL = 20


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
    top = ranked_articles[:K]
    hit = any(a in gold for a in top)
    rr = 0.0
    for rank, a in enumerate(ranked_articles, 1):
        if a in gold:
            rr = 1.0 / rank
            break
    return hit, rr


def verify():
    for item in QUESTIONS:
        print(f"\nQ: {item['q']}  [{item['doc']}]")
        for g in item["gold"][:3]:
            for c in hybrid.CHUNKS:
                if c["doc"] == item["doc"] and c["article"] == g:
                    print(f"  gold {c['label']}: {c['text'][:90]}")
                    break


def main():
    rows = []
    for name, fn in CONFIGS.items():
        hits, rrs = 0, 0.0
        for item in QUESTIONS:
            ranked = articles(fn(normalize_query(item["q"]), LANG[item["doc"]]))
            hit, rr = score(ranked, item["gold"])
            hits += hit
            rrs += rr
        n = len(QUESTIONS)
        rows.append((name, hits / n, rrs / n))
    print(f"\nQuestions: {len(QUESTIONS)}   K = {K}\n")
    print(f"{'config':<20} {'recall@' + str(K):<10} {'MRR':<6}")
    for name, rec, mrr in rows:
        print(f"{name:<20} {rec:<10.3f} {mrr:<6.3f}")


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else main()
