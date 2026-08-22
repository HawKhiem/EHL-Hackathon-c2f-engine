"""Turn a decrypted case folder into one dict the model can read.

{
  "game_id": int,
  "policy": str,            # policy.txt verbatim, in full
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

# Nothing here is length-capped: a policy is ~65k chars at worst, a rounding error against a
# 1M-token window, and cutting it drops the tail where the exclusions and limits live.
# c2f.policy distils the binding clauses and puts them in front of this text.

# Units seen on the invoices. Multi-word units ("flat rate") and a lone dash ("–" = no quantity)
# are allowed; pypdf sometimes glues the quantity to the description ("sink1 pcs"), so the
# whitespace before the quantity is optional.
UNITS = (
    r"pcs|pc|pieces?|units?|hrs|hours?|h|days?|nights?|weeks?|months?|flat rate|lump sum|sets?|pairs?|"
    r"kg|g|l|m|km|m2|m²|sqm|m3|m³|lfm|rolls?|boxes|box|bags?|cans?|litres?|liters?|[–-]"
)
ITEM_RE = re.compile(r"^\s*(\d{1,3})\s+(.+?)\s*(\d+(?:[.,]\d+)?|[–-])\s+(" + UNITS + r")\s*$")
HEADER_RE = re.compile(r"^\s*POS\.?\s+DESCRIPTION", re.I)
STOP_RE = re.compile(r"^\s*(INVOICE|Created on|Page \d|TOTAL|Subtotal|VAT|Notes?)\b", re.I)


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def parse_items(text: str) -> list[dict]:
    """Best-effort parse of the ITEMS table(s). Returns [] if the layout is unfamiliar.

    A PDF may hold several invoices; each has its own POS./DESCRIPTION header and the items are
    numbered continuously across them. Only lines between a header and the next INVOICE/footer
    line are considered, so header fields like "DUE\n5 Mar 2026" can never look like item 5.
    """
    lines = text.splitlines()
    if not any(HEADER_RE.match(ln) for ln in lines):
        return []
    items: list[dict] = []
    in_table = False
    buf: str | None = None  # a line item whose description wraps over several lines

    def flush(b: str) -> bool:
        m = ITEM_RE.match(b)
        if not m:
            return False
        idx, desc, qty, unit = m.groups()
        items.append(
            {
                "index": int(idx),
                "description": desc.strip(),
                "quantity": float(qty.replace(",", ".")) if qty not in "–-" else 0.0,
                "unit": unit,
            }
        )
        return True

    for ln in lines:
        if HEADER_RE.match(ln):
            in_table, buf = True, None
            continue
        if not in_table or not ln.strip():
            continue
        if STOP_RE.match(ln):
            in_table, buf = False, None
            continue
        starts = re.match(r"^\s*(\d{1,3})\s+\S", ln)
        nxt = len(items) + 1
        if buf is None:
            if starts and int(starts.group(1)) == nxt:
                buf = ln.strip()
            # else: noise between items
        elif starts and int(starts.group(1)) == nxt + 1 and not ITEM_RE.match(buf + " " + ln.strip()):
            # the pending item never got a parsable quantity/unit; don't swallow the next item into it
            m = re.match(r"^\s*(\d{1,3})\s+(.+)$", buf)
            items.append({"index": int(m.group(1)), "description": m.group(2).strip(), "quantity": 1.0, "unit": "?"})
            buf = ln.strip()
        else:
            buf += " " + ln.strip()
        if flush(buf):
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
        "policy": policy.strip(),
        "description": desc.strip(),
        "invoice_text": invoice_text.strip(),
        "invoice_meta": parse_meta(invoice_text),
        "items": parse_items(invoice_text),
        "images": images,
    }
