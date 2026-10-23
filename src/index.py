"""Embed all chunks with bge-m3 and store them in Qdrant."""
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

CHUNKS = Path("data/processed/chunks.jsonl")
COLLECTION = "pdp_gdpr"
MODEL = "BAAI/bge-m3"


def main() -> None:
    rows = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    model = SentenceTransformer(MODEL, device="mps")  # use "cpu" if mps errors
    model.max_seq_length = 512

    # prefix with the label so the article number is part of the embedding
    texts = [f"{r['doc']} {r['label']}: {r['text']}" for r in rows]
    vectors = model.encode(
        texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True
    )

    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, r["id"])),
            vector=v.tolist(),
            payload=r,
        )
        for r, v in zip(rows, vectors)
    ]
    client.upsert(COLLECTION, points=points)
    print(f"indexed {len(points)} chunks into '{COLLECTION}'")


if __name__ == "__main__":
    main()