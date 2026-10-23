"""Hybrid retrieval (dense + BM25 via RRF) followed by cross-encoder reranking.
Usage: python -m src.hybrid "your question" [en|id]
"""
import json
import re
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION = "pdp_gdpr"
DEVICE = "mps"  # use "cpu" if mps errors

CHUNKS = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
BY_ID = {c["id"]: c for c in CHUNKS}


def tok(s: str):
    return re.findall(r"\w+", s.lower())


BM25 = BM25Okapi([tok(f"{c['label']} {c['text']}") for c in CHUNKS])
_embedder = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)
_client = QdrantClient(url="http://localhost:6333")


def dense_ids(query: str, n: int, lang: str | None):
    vec = _embedder.encode(query, normalize_embeddings=True).tolist()
    flt = None
    if lang:
        flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=lang))])
    pts = _client.query_points(COLLECTION, query=vec, limit=n, query_filter=flt).points
    return [p.payload["id"] for p in pts]


def bm25_ids(query: str, n: int, lang: str | None):
    scores = BM25.get_scores(tok(query))
    order = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)
    ids = [CHUNKS[i]["id"] for i in order if not lang or CHUNKS[i]["lang"] == lang]
    return ids[:n]


def rrf(rankings, k: int = 60):
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def search(query: str, k: int = 5, lang: str | None = None, pool: int = 20):
    candidates = rrf([dense_ids(query, pool, lang), bm25_ids(query, pool, lang)])[:pool]
    pairs = [(query, f"{BY_ID[c]['label']}: {BY_ID[c]['text']}") for c in candidates]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
    return [(BY_ID[c], float(s)) for c, s in ranked]


if __name__ == "__main__":
    q = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    for chunk, score in search(q, lang=lang):
        print(f"{score:.3f} | {chunk['doc']} {chunk['label']} | {chunk['text'][:100]}")