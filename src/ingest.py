"""Extract and clean text from the raw PDFs."""
import re
from pathlib import Path

import pymupdf

RAW = Path("data/raw")
OUT = Path("data/processed")

# Page headers/footers to drop, per document
NOISE = {
    "gdpr_en": [
        r"^\d{1,2}\.\d{1,2}\.\d{4}$",                 # 4.5.2016
        r"^EN$",
        r"^Official Journal of the European Union$",
        r"^L 119/\d+$",
    ],
        "uu_pdp_id": [
        r"^\W*\s*\d+\s*-?\s*$",                       # ". 3 -", "- 3 -", "3"
        r"^PRESIDEN$",
        r"^REPUBLIK INDONESIA$",
        r"^PRESIDEN REPUBLIK INDONESIA$",
        r"^SK No\s*\d+.*$",
        r"^an$",
    ],
}


def extract(pdf_path: Path) -> str:
    patterns = [re.compile(p) for p in NOISE[pdf_path.stem]]
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            lines = [ln.strip() for ln in page.get_text().splitlines()]
            lines = [ln for ln in lines if ln and not any(p.match(ln) for p in patterns)]
            pages.append("\n".join(lines))
    return "\n".join(pages)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(RAW.glob("*.pdf")):
        text = extract(pdf)
        (OUT / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        print(f"{pdf.name}: {len(text):,} characters")


if __name__ == "__main__":
    main()