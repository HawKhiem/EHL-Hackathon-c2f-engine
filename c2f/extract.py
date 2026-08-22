"""Turn a decrypted case folder into one dict the model can read.

{
  "game_id": int,
  "policy": str,            # policy.txt verbatim, in full
  "description": str,       # description.txt verbatim
  "invoice_text": str,      # full text of invoices.pdf, for the model
  "invoice_meta": {...},    # trade / vendor / date if found
  "items": [],              # always empty: the invoice is NOT line-parsed, the model reads it
  "images": [{"name", "media_type"}],   # listed only - photos are not sent to the model
}
"""

from __future__ import annotations

import re
from pathlib import Path

# Nothing here is length-capped: a policy is ~65k chars at worst, a rounding error against a
# 1M-token window, and cutting it drops the tail where the exclusions and limits live.
# c2f.policy distils the binding clauses and puts them in front of this text.
#
# The invoice is NOT line-parsed. A regex table parse (POS | DESCRIPTION | AMOUNT | UNIT) was
# all-or-nothing and any unfamiliar unit threw the whole invoice away: game 27 billed "68
# lines" of translation, "lines" was not in the unit list, item 1 failed to match and the
# parse discarded all four items. The full pdf text goes to the model and the model reads it.


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)



def parse_meta(text: str) -> dict:
    meta = {}
    for key, pat in {
        "trade": r"TRADE\s*\n?\s*([^\n]+)",
        "invoice_no": r"INVOICE NO\.\s*\n?\s*([^\n]+)",
        "date": r"\bDATE\s*\n?\s*([^\n]+)",
    }.items():
        m = re.search(pat, text)
        if m:
            meta[key] = m.group(1).strip()
    m = re.search(r"FROM\s*\n([^\n]+)", text)
    if m:
        meta["vendor"] = m.group(1).strip()
    return meta


def load_case(case_dir: Path, game_id: int) -> dict:
    case_dir = Path(case_dir)
    policy = (case_dir / "policy.txt").read_text(errors="replace") if (case_dir / "policy.txt").exists() else ""
    desc = (
        (case_dir / "description.txt").read_text(errors="replace")
        if (case_dir / "description.txt").exists()
        else ""
    )
    pdfs = sorted(case_dir.glob("*.pdf"))
    invoice_text = "\n\n".join(pdf_text(p) for p in pdfs)
    images = []
    for p in sorted(case_dir.iterdir()):
        ext = p.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            # Listed, not read: the photos are skipped in the pipeline (c2f.llm sends text
            # only), so the bytes are never loaded or base64-encoded. The names stay in the
            # case dict and the run log so we can see what the case shipped with.
            images.append(
                {
                    "name": p.name,
                    "media_type": "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext[1:]}",
                }
            )
    return {
        "game_id": game_id,
        "policy": policy.strip(),
        "description": desc.strip(),
        "invoice_text": invoice_text.strip(),
        "invoice_meta": parse_meta(invoice_text),
        # No deterministic parse: the model reads the full invoice text above. The key stays
        # so callers that iterate it (run.py, price.py) need no special case.
        "items": [],
        "images": images,
    }
