"""Turn a decrypted case folder into one dict the model can read.

{
  "game_id": int,
  "policy": str,            # policy.txt verbatim, in full
  "description": str,       # description.txt verbatim
  "invoice_text": str,      # full text of invoices.pdf, for the model
  "invoice_meta": {...},    # trade / vendor / date if found
  "items": [{"index", "description", "quantity"?, "unit"?}],  # scheduling/pricing metadata
  "item_labels": {index: description},  # best-effort line labels, for pricing bias + history
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
# The invoice is NOT line-parsed for the model: the full pdf text goes to it and it reads the
# table itself. A regex table parse used to build the model's reading aid and it was
# all-or-nothing - any unfamiliar unit threw the whole invoice away (game 27 billed "68 lines"
# of translation, "lines" was not in the unit list, item 1 failed to match and the parse
# discarded all four items).
#
# item_labels() below is NOT that parse coming back. It recovers the line LABEL only, per item,
# best effort, and nothing downstream of the model depends on it being complete. The labels are
# what c2f.price buckets on to pick a per-category bias, and what c2f.history groups the
# market's accepted t ranges by. Losing them cost more than the reading aid ever did: with no
# label every item bucketed as "other", so games 27-28 priced at the flat global bias instead
# of their category's, and contributed nothing to the market history.

HEADER_RE = re.compile(r"^\s*POS\.?\s+DESCRIPTION", re.I)
STOP_RE = re.compile(r"^\s*(INVOICE|Created on|Page \d|TOTAL|Subtotal|VAT|Notes?)\b", re.I)
# The AMOUNT/UNIT columns as pypdf renders them: a number or a dash ("-" = no quantity),
# then a unit of one or two short words, or another dash. Deliberately NOT a vocabulary of
# known units - that list is what game 27 died on.
# A quantity carries thousands separators as well as a decimal one ("2,412.1kWh" on game
# 17): without the grouped part the strip starts mid-number and leaves "Electricity costs 2,".
_QTY = r"\d+(?:[.,]\d{3})*(?:[.,]\d+)?"
_DASH = "[–—-]"  # en dash, em dash, hyphen - the invoices use all three for "no quantity"
_UNIT = r"(?:[A-Za-z][A-Za-z²³]{0,11}(?:\s+[A-Za-z]{1,8})?|" + _DASH + ")"
_QTY_UNIT = "(?:" + _QTY + "|" + _DASH + r")\s*" + _UNIT
# A line that is ONLY a quantity and unit is the tail of the item above, wrapped onto its own
# line - never a new item. Game 28 item 1 billed "12 hrs" that way, and reading it as item 12
# would open a phantom item AND swallow the real item 2 as its continuation.
QTY_LINE_RE = re.compile(r"^\s*" + _QTY_UNIT + r"\s*$")
# ...and the same columns glued to the end of the description ("kitchen sink1 pcs").
TAIL_RE = re.compile(r"\s*" + _QTY_UNIT + r"?\s*$")
ITEM_START_RE = re.compile(r"^(\d{1,3})\s+(\S.*)$")


def _label(parts: list[str]) -> str:
    """Join a wrapped description and drop the quantity/unit columns off the end."""
    text = " ".join(parts).strip()
    return TAIL_RE.sub("", text).strip() or text


def _item_rows(text: str) -> dict[int, list[str]]:
    """POS number -> raw wrapped description/quantity columns.

    Best effort and per item: an item that cannot be read is the only one lost. A pdf may hold
    several invoices, each with its own POS./DESCRIPTION header, numbered continuously across
    them; numbering runs upward but is not gap-free (game 11 went 1..11, 13..23), so a new item
    is ANY index above the last one. Header fields between invoices ("DUE / 5 Mar 2026") sit
    outside the table and are never read as items.
    """
    lines = text.splitlines()
    if not any(HEADER_RE.match(ln) for ln in lines):
        return {}
    rows: dict[int, list[str]] = {}
    in_table = False
    idx: int | None = None
    buf: list[str] = []
    last = 0

    def flush() -> None:
        nonlocal idx, buf
        if idx is not None and buf:
            rows[idx] = buf
        idx, buf = None, []

    for ln in lines:
        if HEADER_RE.match(ln):
            flush()
            in_table = True
            continue
        if not in_table or not ln.strip():
            continue
        if STOP_RE.match(ln):
            flush()
            in_table = False
            continue
        s = ln.strip()
        m = ITEM_START_RE.match(s)
        if m and int(m.group(1)) > last and not QTY_LINE_RE.match(s):
            flush()
            idx = last = int(m.group(1))
            buf = [m.group(2)]
        elif idx is not None:
            buf.append(s)
    flush()
    return rows


def item_labels(text: str) -> dict[int, str]:
    """POS number -> line description, for every item table in the pdf text."""
    return {i: label for i, parts in _item_rows(text).items() if (label := _label(parts))}


QTY_CAPTURE_RE = re.compile(r"(" + _QTY + r")\s*(" + _UNIT + r")\s*$", re.I)


def item_quantities(text: str) -> dict[int, tuple[float, str]]:
    """Best-effort POS -> (quantity, unit); pricing never depends on missing entries."""
    out = {}
    for i, parts in _item_rows(text).items():
        m = QTY_CAPTURE_RE.search(" ".join(parts).strip())
        if not m:
            continue
        raw = m.group(1)
        if "," in raw and "." in raw:
            raw = raw.replace(",", "")
        elif raw.count(",") == 1:
            left, right = raw.split(",")
            raw = left + ("" if len(right) == 3 else ".") + right
        out[i] = (float(raw), " ".join(m.group(2).lower().split()))
    return out



def case_labels(case: dict) -> dict[int, str]:
    """Line labels for a case dict, whatever vintage of run log it came out of.

    Read by everything that needs the CATEGORY of an item after the fact - c2f.price for the
    per-bucket bias, c2f.calibrate/accuracy/deviation/history for grouping. Three sources, best
    first: the labels load_case now stores (json has turned the int keys into strings by the
    time a log is read back), else a fresh recovery from the invoice text every log keeps, else
    the old deterministic parse in logs that still carry one.

    The recovery is tried BEFORE the stored parse on purpose: measured over every run log we
    have it reproduces 313 of that parse's 315 labels, is cleaner on the other two, and reads
    23 items the parse dropped outright (game 4 items 5-15, game 11 items 13-23).
    """
    out: dict[int, str] = {}
    for k, v in (case.get("item_labels") or {}).items():
        try:
            out[int(k)] = str(v or "")
        except (TypeError, ValueError):
            continue
    if out:
        return out
    out = item_labels(case.get("invoice_text") or "")
    if out:
        return out
    for it in case.get("items") or []:
        try:
            out[int(it["index"])] = str(it.get("description") or "")
        except (KeyError, TypeError, ValueError):
            continue
    return out


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
    labels = item_labels(invoice_text)
    quantities = item_quantities(invoice_text)
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
        # The POS numbers item_labels recovered. NOT a reading aid for the model - the model
        # still reads the full invoice_text above and the prompt carries no parsed table.
        # This list exists so c2f.run can split a long invoice into parallel calls.
        #
        # It used to be left empty on the grounds that a chunk plan should not ride on a
        # best-effort recovery. The measurements since say otherwise, on both halves:
        #  - The recovery is not in fact best-effort in practice: it reproduces the POS list
        #    exactly on all 37 stored invoices, including the two that broke the old parser
        #    (game 11's 1..11,13..23 and game 27's "68 lines").
        #  - Leaving it empty is not the safe side. One un-chunked call is all-or-nothing and
        #    its cost scales with the invoice: games 29/30/32/33 (4-7 items) answered in
        #    6.5-9.2 s, but game 31 (18 items) took 32.8 s and its submission came back 403
        #    GAME ALREADY ENDED, and game 35 (20 items) took 37.9 s and only just landed.
        #    Chunked, the round's latency is the slowest chunk (case 35 measured 33.9 s -> 26.8 s)
        #    and - the part that actually matters - c2f.run submits after EVERY chunk, so a
        #    slow chunk costs its own items instead of shipping an empty board.
        # A wrong or short list still cannot lose a line: chunk 0 sweeps for POS numbers
        # missing from its list, and merge_estimates unions the model's indices with these.
        "items": [
            {"index": i, "description": d, **(
                {"quantity": quantities[i][0], "unit": quantities[i][1]} if i in quantities else {}
            )}
            for i, d in sorted(labels.items())
        ],
        "item_labels": labels,
        "images": images,
    }
