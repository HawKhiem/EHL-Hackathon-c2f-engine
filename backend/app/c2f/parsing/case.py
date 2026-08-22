"""Load a decrypted case directory into a `CaseBundle`.

Tolerant by design. A missing `policy.txt` costs us accuracy; a raised exception
costs us the round. Every file is optional except the invoice, and even a
zero-item invoice returns a bundle rather than failing.

Policy and description are **not parsed** - they are read verbatim and handed to
the model as-is. Clause extraction would be a lossy pre-filter in front of the
one component able to read the documents properly, and case 0 shows the cost:
its decisive sentence is section 4's "market value at the time of the theft",
which an extractor tuned for exclusions would drop.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.c2f.inference.analyse import CaseBundle
from app.c2f.parsing.invoice import parse_invoice

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
#: Guards against a pathological policy blowing the prompt budget.
MAX_TEXT_CHARS: int = 40_000


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:MAX_TEXT_CHARS]
    except OSError:
        log.warning("c2f could not read %s", path)
        return ""


def load_case(case_dir: str | Path, *, case_id: str | None = None) -> CaseBundle:
    """Read one decrypted case folder.

    Files are matched by stem so a `policy.md` or `beschreibung.txt` still lands
    somewhere sensible rather than being silently dropped.
    """
    directory = Path(case_dir)
    policy = ""
    description = ""
    invoice_path: Path | None = None
    images: list[str] = []

    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if not path.is_file():
            continue
        stem = path.stem.lower()
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # Prefer a file that actually says "invoice" when there are several.
            if invoice_path is None or "invoice" in stem:
                invoice_path = path
        elif suffix in _IMAGE_SUFFIXES:
            images.append(str(path))
        elif "polic" in stem or "bedingung" in stem:
            policy = _read_text(path)
        elif "descr" in stem or "damage" in stem or "beschreib" in stem or "schaden" in stem:
            description = _read_text(path)
        elif suffix in {".txt", ".md"} and not description:
            description = _read_text(path)

    items = parse_invoice(invoice_path) if invoice_path else []
    if not items:
        log.error("c2f parsed zero line items from %s", invoice_path or directory)

    return CaseBundle(
        case_id=case_id or directory.name,
        items=items,
        policy=policy,
        description=description,
        image_paths=images,
    )
