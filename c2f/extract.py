"""Turn a decrypted case folder into one dict the model can read.

{
  "game_id": int,
  "policy": str,            # policy.txt verbatim (length-capped)
  "description": str,       # description.txt verbatim
  "invoice_text": str,      # full text of invoices.pdf, for the model
  "invoice_meta": {...},    # trade / vendor / date if found
  "items": [{"index", "description", "quantity", "unit"}],  # deterministic parse, may be empty
  "images": [{"name", "media_type", "b64"}],
}
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

MAX_CHARS = 30_000

ITEM_RE = re.compile(r"^\s*(\d{1,3})\s+(.+?)\s+(\d+(?:[.,]\d+)?|[–-])\s+(\S+)\s*$")
HEADER_RE = re.compile(r"^\s*POS\.?\s+DESCRIPTION", re.I)
STOP_RE = re.compile(r"^\s*(INVOICE|Created on|Page \d|TOTAL|Subtotal|VAT|Notes?)\b", re.I)


def _cap(s: str) -> str:
    s = s.strip()
    if len(s) > MAX_CHARS:
        return s[:MAX_CHARS] + "\n[... truncated ...]"
    return s


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def parse_items(text: str) -> list[dict]:
    """Best-effort parse of the ITEMS table. Returns [] if the layout is unfamiliar."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if HEADER_RE.match(ln)), None)
    if start is None:
        return []
    items: list[dict] = []
    buf: str | None = None  # a line item whose description wraps over several lines
    for ln in lines[start + 1 :]:
        if not ln.strip():
            continue
        starts_next = re.match(r"^\s*(\d{1,3})\s+\S", ln)
        if buf is None:
            if starts_next and int(starts_next.group(1)) == len(items) + 1:
                buf = ln.strip()
            else:
                continue  # noise between items (e.g. footer text)
        else:
            buf += " " + ln.strip()
        m = ITEM_RE.match(buf)
        if m:
            idx, desc, qty, unit = m.groups()
            items.append(
                {
                    "index": int(idx),
                    "description": desc.strip(),
                    "quantity": float(qty.replace(",", ".")) if qty not in "–-" else 0.0,
                    "unit": unit,
                }
            )
            buf = None
    # indices must be 1..n and unique, else distrust the parse
    idxs = [it["index"] for it in items]
    if not idxs or sorted(idxs) != list(range(1, len(idxs) + 1)):
        return []
    return items


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
            images.append(
                {
                    "name": p.name,
                    "media_type": "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext[1:]}",
                    "b64": base64.b64encode(p.read_bytes()).decode(),
                }
            )
    return {
        "game_id": game_id,
        "policy": _cap(policy),
        "description": _cap(desc),
        "invoice_text": _cap(invoice_text),
        "invoice_meta": parse_meta(invoice_text),
        "items": parse_items(invoice_text),
        "images": images,
    }
