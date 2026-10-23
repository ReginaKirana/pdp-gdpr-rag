"""Citation check on generated answers (no LLM judge, fully deterministic).

For every question it generates an answer and measures:
  - citation validity : cited articles that are really among the retrieved passages
  - gold cited        : the answer cites at least one gold article
  - over-refusal      : an in-scope question that was refused
  - uncited answers   : answered, but with no citation at all

Usage:
  python -m eval.run_citations --limit 6                 # quick smoke test
  python -m eval.run_citations --file eval/questions_heldout.jsonl
Needs Qdrant and Ollama running (run it on the host, not in Docker).
"""
import argparse
import json
import re
import time
from pathlib import Path

from src.generate import REFUSAL, answer

CITE = re.compile(r"\[(GDPR|UU PDP)\s+([^\]]+)\]")
NUM = re.compile(r"(?:Article|Pasal)\s+(\d+)")


def citations(text: str):
    """[('GDPR', 33), ('UU PDP', 46), ...] from citations like [GDPR Article 33(1)]."""
    out = []
    for m in CITE.finditer(text):
        for n in NUM.findall(m.group(2)):
            out.append((m.group(1), int(n)))
    return out


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval/questions_heldout.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="eval/results_citations.jsonl")
    args = ap.parse_args()

    questions = load(args.file)[: args.limit]
    records = []
    for i, item in enumerate(questions, 1):
        t0 = time.perf_counter()
        text, hits = answer(item["q"])
        secs = time.perf_counter() - t0

        retrieved = {(doc, c["article"]) for doc, items in hits.items() for c, _ in items}
        cites = citations(text)
        refused = text.strip() == REFUSAL
        invalid = [c for c in cites if c not in retrieved]
        gold_cited = any((item["doc"], g) in cites for g in item["gold"])
        records.append({
            "q": item["q"], "doc": item["doc"], "gold": item["gold"],
            "answer": text, "cites": cites, "invalid": invalid,
            "refused": refused, "gold_cited": gold_cited, "seconds": round(secs, 1),
        })
        print(f"[{i}/{len(questions)}] {secs:5.1f}s  cites={len(cites)} invalid={len(invalid)} "
              f"refused={refused} gold_cited={gold_cited} | {item['q'][:60]}")

    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )

    n = len(records)
    answered = [r for r in records if not r["refused"]]
    total = sum(len(r["cites"]) for r in records)
    invalid = sum(len(r["invalid"]) for r in records)
    print(f"\nQuestions: {n}   (answers saved to {args.out})")
    print(f"Over-refusals (in-scope question refused): {n - len(answered)}")
    print(f"Answers with no citation: {sum(1 for r in answered if not r['cites'])}")
    if total:
        print(f"Citation validity: {total - invalid}/{total} = {(total - invalid) / total:.3f}")
    if answered:
        gc = sum(r["gold_cited"] for r in answered)
        print(f"Gold article cited: {gc}/{len(answered)} = {gc / len(answered):.3f}")
    bad = [r for r in records if r["invalid"]]
    if bad:
        print("\nInvalid citations (cited but not among the retrieved passages):")
        for r in bad:
            print(f"  {r['invalid']} | {r['q'][:70]}")


if __name__ == "__main__":
    main()
