"""Turn a decrypted case folder into one dict the model can read.

{
  "game_id": int,
  "policy": str,            # policy.txt verbatim, in full
  "description": str,       # description.txt verbatim
  "invoice_text": str,      # full text of invoices.pdf, for the model - AUTHORITATIVE
  "invoice_meta": {...},    # trade / vendor / date if found
  "items": [{"index", "description"}],  # POS numbers only, advisory, may be empty
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
# THE INVOICE TEXT IS AUTHORITATIVE. The model reads the full pypdf text and decides for
# itself what the line items are and what they cost; the prompt carries no parsed table.
# What is parsed here is the POS number and the words after it - nothing that can be
# "unfamiliar". The old parser also demanded a quantity and a unit from a fixed list and
# threw the WHOLE invoice away when one line did not match: game 27 billed "68 lines" of
# translation, "lines" was not in UNITS, item 1 failed, and all four items were discarded.
#
# Three consumers need the POS numbers, and none of them is the model's reading:
#   * c2f.run splits a long invoice into parallel calls by POS number. Games 10 and 15 both
#     shipped fast-pass prices because ONE whole-invoice call missed the 60 s deadline.
#   * c2f.run.merge_estimates spots a POS number the model silently skipped and prices it
#     as unknown instead of leaving the line off the board.
#   * c2f.price picks a per-category bias from the invoice wording - we under-price material
#     and labour while over-pricing drying, so one global multiplier is wrong for both.
#
# The parse is ADVISORY and degrades instead of failing. A partial parse costs a chunk plan,
# never a line item: chunk 0 sweeps for POS numbers missing from its list (c2f.llm), and
# merge_estimates unions the model's indices with the parsed ones. An empty parse just means
# one un-chunked call over the full text - exactly what we shipped for games 27-29.

HEADER_RE = re.compile(r"^\s*POS\.?\s+DESCRIPTION", re.I)
STOP_RE = re.compile(r"^\s*(INVOICE|Created on|Page \d|TOTAL|Subtotal|VAT|Notes?)\b", re.I)
#: a POS number and the text after it. No quantity, no unit, nothing to be unfamiliar with.
POS_RE = re.compile(r"^\s*(\d{1,3})\s+(\S.*?)\s*$")
#: cosmetic tail strip: "... rug 1 pcs" -> "... rug", "... (metered) - -" -> "... (metered)".
#: Never load-bearing - if it does not match, the units ride along in the description.
TAIL_RE = re.compile(r"\s+(?:\d+(?:[.,]\d+)?|[\u2013\u2014-])\s+\S{1,10}(?:\s+\S{1,6})?\s*$")
#: Biggest jump in POS numbering we will believe. Invoices DO skip numbers - game 11 went
#: 1..11, 13..23 - but a wrapped description starting "230 V cable" is not item 230.
MAX_GAP = 20
#: Fewest whitespace tokens after the POS number for a line to be a NEW item rather than the
#: wrapped tail of the one above. This is the whole defence against the trap the old parser
#: used its unit list for, and it needs no unit list: a real item line carries a description
#: AND a quantity AND a unit ("2 Vehicle costs 1 pcs" -> 4), while the orphaned quantity of a
#: wrapped description carries only a number and a unit ("12 hrs" -> 1, "1 flat rate" -> 2).
#: Game 28 item 1's quantity wrapped onto its own line as "12 hrs"; read as item 12 it
#: swallowed the five items under it. A false item is far worse than a missed one - it eats
#: its followers, where a miss is recovered by the chunk-0 sweep - so the test is deliberately
#: strict and errs toward "continuation".
MIN_ITEM_TOKENS = 3


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _clean(desc: str) -> str:
    stripped = TAIL_RE.sub("", desc).strip()
    return stripped if len(stripped) >= 2 else desc.strip()


def parse_pos(text: str) -> list[dict]:
    """POS number + description for each line item. [] when the layout is unrecognisable.

    A PDF may hold several invoices; each has its own POS./DESCRIPTION header and the items
    are numbered continuously across them. Only lines between a header and the next
    INVOICE/footer line are considered, so a header field like "DUE\n5 Mar 2026" can never
    look like item 5.

    Inside a table region every line either opens a new item (it starts with a number above
    the last one, by no more than MAX_GAP) or continues the description of the current one.
    That is the whole grammar: a description that wraps over three lines and ends in
    "1 flat rate" needs no special case, and neither does a unit nobody has seen before.
    """
    items: list[dict] = []
    in_table = False
    for ln in text.splitlines():
        if HEADER_RE.match(ln):
            in_table = True
            continue
        if not in_table or not ln.strip():
            continue
        if STOP_RE.match(ln):
            in_table = False
            continue
        m = POS_RE.match(ln)
        last = items[-1]["index"] if items else 0
        if (m and last < int(m.group(1)) <= last + MAX_GAP
                and len(m.group(2).split()) >= MIN_ITEM_TOKENS):
            items.append({"index": int(m.group(1)), "description": m.group(2).strip()})
        elif items:
            items[-1]["description"] += " " + ln.strip()
        # else: noise before the first item of the first table
    # Numbering starts at 1 and rises (rising is guaranteed above). A parse that starts
    # anywhere else means we misread the layout, and no parse is the safe answer: the model
    # still gets the full text, we just lose the chunk plan.
    if not items or items[0]["index"] != 1:
        return []
    for it in items:
        it["description"] = _clean(it["description"])
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
        # POS numbers only, and only as a reading aid for c2f.run and c2f.price - the model
        # reads invoice_text above. [] when the layout is unrecognisable; see parse_pos.
        "items": parse_pos(invoice_text),
        "images": images,
    }
