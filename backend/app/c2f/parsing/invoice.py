"""Pull line items out of `invoices.pdf`.

The submission API addresses line items by their 1-based `index`, which is the
POS column on the invoice. Getting that mapping wrong misprices every item in
the round, so the parser is deliberately literal: it trusts the POS column and
never renumbers.

Rows look like

    POS. DESCRIPTION            AMOUNT UNIT TOTAL
    1    New Bike                    1 unit

with TOTAL empty - the prices are the thing we are being asked for. Parsed
right-to-left, because a description may itself contain numbers ("26 inch
wheel") while the trailing `amount unit` pair is positionally reliable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.c2f.models import LineItem

log = logging.getLogger(__name__)

#: `1  New Bike  1 unit`  ->  (pos, description, amount, unit)
#: The description is greedy so the LAST number on the row is the amount.
_ROW = re.compile(r"^(\d{1,3})[.)]?\s+(.+?)\s+([\d.,]+)\s*([A-Za-z%]{0,12})\s*$")
#: A row with no amount column at all. Quantity then defaults to 1.
_BARE_ROW = re.compile(r"^(\d{1,3})[.)]?\s+(.+?)\s*$")

_HEADER = re.compile(r"POS\.?\s+DESCRIPTION", re.IGNORECASE)
_STOP = re.compile(
    r"^\s*(created on|page \d|invoice$|subtotal|total\s|net\s|vat|ust|mwst|"
    r"grand total|amount due|thank you)",
    re.IGNORECASE,
)
#: Column headers leak into the text layer as a row; never an item.
_NOISE = re.compile(r"^\s*(pos\.?|description|amount|unit|total|items)\b", re.IGNORECASE)


def parse_amount(raw: str) -> float:
    """`1`, `1.5`, `1,5`, `1.200,50` -> float. Returns 1.0 on anything unreadable."""
    text = raw.strip()
    if not text:
        return 1.0
    if "," in text and "." in text:
        # Whichever separator comes last is the decimal point.
        text = (
            text.replace(".", "").replace(",", ".")
            if text.rindex(",") > text.rindex(".")
            else text.replace(",", "")
        )
    elif "," in text:
        # A single comma is a decimal comma unless it splits thousands.
        head, _, tail = text.rpartition(",")
        text = f"{head}.{tail}" if len(tail) != 3 or not head else f"{head}{tail}"
    try:
        value = float(text)
    except ValueError:
        return 1.0
    return value if value > 0 else 1.0


def extract_text(pdf_path: str | Path) -> str:
    """Every page's text layer, concatenated. Empty string if unreadable."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:  # noqa: BLE001 - a broken PDF must not end the round
        log.exception("c2f could not read %s", pdf_path)
        return ""


def parse_line_items(text: str) -> list[LineItem]:
    """Line items in invoice order, keyed by their POS index.

    Only rows after the `POS. DESCRIPTION` header are considered, so addresses
    and invoice numbers cannot be mistaken for items.
    """
    items: list[LineItem] = []
    seen: set[int] = set()
    in_table = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _HEADER.search(line):
            in_table = True
            continue
        if not in_table:
            continue
        if _STOP.match(line):
            in_table = False
            continue
        if _NOISE.match(line):
            continue

        match = _ROW.match(line)
        if match:
            pos, description, amount, unit = match.groups()
            quantity = parse_amount(amount)
        else:
            match = _BARE_ROW.match(line)
            if not match:
                continue
            pos, description = match.groups()
            quantity, unit = 1.0, ""

        index = int(pos)
        description = description.strip(" .-–—")
        if index in seen or not description:
            continue
        seen.add(index)

        items.append(
            LineItem(
                item_id=str(index),
                index=index,
                description=description,
                quantity=quantity,
                unit=unit.strip() or None,
                raw_text=line,
            )
        )

    items.sort(key=lambda item: item.index)
    return items


def parse_invoice(pdf_path: str | Path) -> list[LineItem]:
    return parse_line_items(extract_text(pdf_path))
