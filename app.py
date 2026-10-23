"""Web app: Gradio UI + FastAPI endpoints.
Run:  python app.py        then open http://localhost:7860
API:  POST /api/ask {"question": "..."}      GET /health
"""
import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generate import answer

TITLE = "PDP-GDPR RAG"
DESCRIPTION = (
    "Ask questions about the EU GDPR and Indonesia's Personal Data Protection Law "
    "(UU PDP No. 27/2022). Answers cite the articles they come from. "
    "You can ask in English or Indonesian. This is a research aid, not legal advice."
)
EXAMPLES = [
    "How does Indonesia's PDP Law differ from GDPR on data breach notification?",
    "Apa hak subjek data pribadi menurut UU PDP?",
    "What conditions must consent meet to be valid under the GDPR?",
    "Bagaimana aturan pemasangan CCTV di tempat umum?",
    "What is the penalty for tax evasion in Thailand?",
]

api = FastAPI(title=TITLE)


class Ask(BaseModel):
    question: str


def _sources(hits_by_doc: dict):
    return [
        {
            "doc": doc,
            "label": chunk["label"],
            "score": round(score, 3),
            "text": chunk["text"],
        }
        for doc, hits in hits_by_doc.items()
        for chunk, score in hits
    ]


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/api/ask")
def ask(body: Ask):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")
    try:
        text, hits = answer(question)
    except Exception as exc:  # Qdrant or Ollama not running
        raise HTTPException(status_code=503, detail=f"backend unavailable: {exc}")
    return {"answer": text, "sources": _sources(hits)}


def ask_ui(question: str):
    question = (question or "").strip()
    if not question:
        return "Please type a question.", ""
    try:
        text, hits = answer(question)
    except Exception as exc:
        return (
            "The backend is not reachable. Make sure Qdrant (Docker) and Ollama are "
            f"running.\n\n`{exc}`",
            "",
        )
    lines = [
        "Scores are cosine similarities within each law; they are not comparable "
        "between GDPR and UU PDP.\n"
    ]
    for s in _sources(hits):
        snippet = s["text"][:350].replace("\n", " ")
        lines.append(f"**{s['doc']} {s['label']}** (score {s['score']})\n> {snippet}...\n")
    return text, "\n".join(lines)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    question = gr.Textbox(label="Question", lines=2, placeholder="Ask about GDPR or UU PDP...")
    button = gr.Button("Ask", variant="primary")
    output = gr.Markdown(label="Answer")
    with gr.Accordion("Retrieved passages", open=False):
        sources = gr.Markdown()
    gr.Examples(examples=EXAMPLES, inputs=question)
    button.click(ask_ui, inputs=question, outputs=[output, sources])
    question.submit(ask_ui, inputs=question, outputs=[output, sources])

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)