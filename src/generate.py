"""Answer a question from retrieved legal text, with citations (local LLM via Ollama).
Usage: python -m src.generate "your question"
"""
import os
import sys

import httpx

from src.hybrid import detect_laws, search_both

REFUSAL = "I could not find this in the provided texts."

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")  # "qwen2.5:3b" on an 8 GB Mac

SYSTEM = """You are a legal research assistant for two laws: the EU GDPR (English) \
and Indonesia's Personal Data Protection Law, UU PDP (Indonesian).

Rules:
- Answer ONLY from the context passages. Never use outside knowledge.
- Always answer in English, even when a passage is in Indonesian.
- Cite every claim like [GDPR Article 33] or [UU PDP Pasal 46].
- When the question asks for a list (rights, obligations, requirements), cover every relevant item in the passages, not just one.
- If the passages do not contain the answer, reply with exactly this sentence and nothing else: "I could not find this in the provided texts." Never add that sentence after an answer. Do not guess.
- Mention each article only once; never repeat the same point.
- If a question involves both laws, cover each law separately, then summarize differences.
- You are not a lawyer; do not give legal advice."""


def build_context(hits_by_doc: dict) -> str:
    parts = []
    for doc, hits in hits_by_doc.items():
        for chunk, _score in hits:
            parts.append(f"[{doc} {chunk['label']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def clean_answer(text: str) -> str:
    """The refusal sentence is only valid as the entire reply. If the model
    appended it to a real answer, drop it."""
    text = text.strip()
    if text != REFUSAL and REFUSAL in text:
        text = text.replace(REFUSAL, "").strip()
    return text


def answer(question: str, method: str = "dense", k: int | None = None):
    if k is None:
        # one law named -> only that law is searched, so use the whole passage budget on it
        k = 12 if len(detect_laws(question)) == 1 else 8
    hits = search_both(question, k=k, method=method)
    context = build_context(hits)
    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
                },
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return clean_answer(resp.json()["message"]["content"]), hits


if __name__ == "__main__":
    text, hits = answer(sys.argv[1])
    print(text)
    print("\nSources retrieved:")
    for doc, items in hits.items():
        print(f"  {doc}: " + ", ".join(c["label"] for c, _ in items))
