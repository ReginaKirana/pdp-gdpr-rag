"""Split each law into one chunk per Article / Pasal, with metadata."""
import json
import re
from pathlib import Path

PROCESSED = Path("data/processed")
MAX_CHARS = 2000  # long articles (e.g. definitions) are split further

DOCS = {
    "gdpr_en": {
        "name": "GDPR", "lang": "en",
        "start": r"HAVE ADOPTED THIS REGULATION",   # skip the recitals
        "end": None,
        "unit": re.compile(r"^Article (\d+)\s*$", re.M),
        "chapter": re.compile(r"^CHAPTER ([IVX]+)\s*$", re.M),
        "label": "Article",
    },
    "uu_pdp_id": {
        "name": "UU PDP", "lang": "id",
        "start": None,
        "end": r"^PENJELASAN\b",                    # skip the elucidation section
        "unit": re.compile(r"^Pasa[l1I]\s+(\d+)\s*$", re.M),
        "chapter": re.compile(r"^BAB ([IVX]+)\s*$", re.M),
        "label": "Pasal",
    },
}


def normalize(text: str) -> str:
    # join wrapped lines, but keep list items like (a), (1), a., 1. on their own line
    text = re.sub(r"\n(?!\(\w+\)|\d+\.\s|[a-z]\.\s)", " ", text)
    text = re.sub(r"\b(to|of|in|on|by|for)(the|a|an)\b", r"\1 \2", text)
    text = re.sub(r"\s'\s", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def split_long(text: str):
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n"):
        if cur and len(cur) + len(para) > MAX_CHARS:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_doc(key: str, cfg: dict):
    text = (PROCESSED / f"{key}.txt").read_text(encoding="utf-8")

    if cfg["start"]:
        m = re.search(cfg["start"], text, re.M)
        text = text[m.end():] if m else text
    if cfg["end"]:
        m = re.search(cfg["end"], text, re.M)
        text = text[:m.start()] if m else text

    chapters = [(m.start(), m.group(1)) for m in cfg["chapter"].finditer(text)]
    units = list(cfg["unit"].finditer(text))

    chunks = []
    for i, m in enumerate(units):
        end = units[i + 1].start() if i + 1 < len(units) else len(text)
        body = text[m.end():end]
        # drop a trailing chapter heading that belongs to the next article
        body = re.split(cfg["chapter"].pattern, body, flags=re.M)[0]
        body = normalize(body)
        chapter = next((c for pos, c in reversed(chapters) if pos < m.start()), None)
        for j, part in enumerate(split_long(body)):
            chunks.append({
                "id": f"{key}-{cfg['label'].lower()}{m.group(1)}-{j}",
                "doc": cfg["name"],
                "lang": cfg["lang"],
                "article": int(m.group(1)),
                "label": f"{cfg['label']} {m.group(1)}",
                "chapter": chapter,
                "text": part,
            })
    return chunks


def main() -> None:
    all_chunks = []
    for key, cfg in DOCS.items():
        chunks = chunk_doc(key, cfg)
        numbers = sorted({c["article"] for c in chunks})
        print(f"{cfg['name']}: {len(numbers)} {cfg['label']}s -> {len(chunks)} chunks")
        expected = 99 if key == "gdpr_en" else 76
        missing = [n for n in range(1, expected + 1) if n not in numbers]
        if missing:
            print(f"  WARNING: missing {cfg['label']}s: {missing}")
        all_chunks += chunks
    out = PROCESSED / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"saved {len(all_chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()