"""Dense retrieval test: python -m src.retrieve "your question" [en|id]"""
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

COLLECTION = "pdp_gdpr"
_model = None


def search(query: str, k: int = 5, lang: str | None = None):
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-m3", device="mps")  # "cpu" if mps errors
    vec = _model.encode(query, normalize_embeddings=True).tolist()
    flt = None
    if lang:
        flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=lang))])
    client = QdrantClient(url="http://localhost:6333")
    return client.query_points(
        COLLECTION, query=vec, limit=k, query_filter=flt, with_payload=True
    ).points


if __name__ == "__main__":
    q = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    for p in search(q, lang=lang):
        pl = p.payload
        print(f"{p.score:.3f} | {pl['doc']} {pl['label']} | {pl['text'][:120]}")