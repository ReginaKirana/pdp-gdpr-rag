"""Retrieval over GDPR + UU PDP: dense (default), BM25 hybrid, and reranking.
Models load lazily, so importing this module is cheap.

Usage: python -m src.hybrid "your question" [en|id] [dense|rerank]
"""
import json
import os
import re
import sys
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION = "pdp_gdpr"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = os.environ.get("DEVICE") or _device()

CHUNKS = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
BY_ID = {c["id"]: c for c in CHUNKS}


def tok(s: str):
    return re.findall(r"\w+", s.lower())


_ID = re.compile(r"\b(?:Indonesia(?:'s|n)?\s+)?(?:PDP|Personal Data Protection)\s+Law\b", re.I)
_UU = re.compile(r"\bUU\s*PDP\b", re.I)
_UU_PREP = re.compile(r"\b(?:menurut|berdasarkan|dalam|pada|di)\s+UU\s*PDP\b", re.I)
_EN = re.compile(r"\bGDPR\b|\bGeneral Data Protection Regulation\b", re.I)
_IN_ID = re.compile(r"\s+(?:in|within)\s+Indonesia\b", re.I)


def detect_laws(query: str) -> set:
    """Which law(s) does the question name explicitly?"""
    laws = set()
    if _EN.search(query):
        laws.add("GDPR")
    if _ID.search(query) or _UU.search(query):
        laws.add("UU PDP")
    return laws


def normalize_query(query: str) -> str:
    """The language filter already selects the law, so its name is only noise
    (it pulls in articles that mention 'Indonesia' or the law itself). Leave GDPR
    and comparison questions untouched."""
    if _EN.search(query):
        return query
    q = _UU_PREP.sub("", query)
    q = _UU.sub("undang-undang", q)
    q = _ID.sub("the law", q)
    q = _IN_ID.sub("", q)
    return re.sub(r"\s+([?.,!])", r"\1", q).strip()


@lru_cache(maxsize=None)
def _embedder():
    return SentenceTransformer("BAAI/bge-m3", device=DEVICE)


@lru_cache(maxsize=None)
def _reranker():
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)


@lru_cache(maxsize=None)
def _client():
    return QdrantClient(url=QDRANT_URL)


@lru_cache(maxsize=None)
def _bm25():
    return BM25Okapi([tok(f"{c['label']} {c['text']}") for c in CHUNKS])


def dense_hits(query: str, n: int, lang: str | None):
    vec = _embedder().encode(query, normalize_embeddings=True).tolist()
    flt = None
    if lang:
        flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=lang))])
    return _client().query_points(
        COLLECTION, query=vec, limit=n, query_filter=flt, with_payload=True
    ).points


def dense_ids(query: str, n: int, lang: str | None):
    return [p.payload["id"] for p in dense_hits(query, n, lang)]


def bm25_ids(query: str, n: int, lang: str | None):
    scores = _bm25().get_scores(tok(query))
    order = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)
    ids = [CHUNKS[i]["id"] for i in order if not lang or CHUNKS[i]["lang"] == lang]
    return ids[:n]


def rrf(rankings, k: int = 60):
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def search_dense(query: str, k: int = 5, lang: str | None = None):
    """Default retriever: bge-m3 vector search. Returns [(chunk, cosine_score)]."""
    query = normalize_query(query)
    return [(BY_ID[p.payload["id"]], float(p.score)) for p in dense_hits(query, k, lang)]


def search(query: str, k: int = 5, lang: str | None = None, pool: int = 20):
    """Hybrid (dense + BM25 via RRF) followed by cross-encoder reranking."""
    query = normalize_query(query)
    candidates = rrf([dense_ids(query, pool, lang), bm25_ids(query, pool, lang)])[:pool]
    pairs = [(query, f"{BY_ID[c]['label']}: {BY_ID[c]['text']}") for c in candidates]
    scores = _reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
    return [(BY_ID[c], float(s)) for c, s in ranked]


def search_both(query: str, k: int = 3, method: str = "dense", route: bool = True):
    """Top-k from each law separately; scores are not comparable across languages.
    If the question names exactly one law, only that law is searched."""
    fn = search_dense if method == "dense" else search
    targets = {"GDPR": "en", "UU PDP": "id"}
    laws = detect_laws(query) if route else set()
    if len(laws) == 1:
        targets = {doc: lang for doc, lang in targets.items() if doc in laws}
    return {doc: fn(query, k=k, lang=lang) for doc, lang in targets.items()}


if __name__ == "__main__":
    q = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    method = sys.argv[3] if len(sys.argv) > 3 else "dense"
    fn = search_dense if method == "dense" else search
    for chunk, score in fn(q, lang=lang):
        print(f"{score:.3f} | {chunk['doc']} {chunk['label']} | {chunk['text'][:100]}")
