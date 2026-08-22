"""Extraction of the three raw text sources in a case: the policy, the damage description, and
the invoice (pypdf gives us the PDF's text layer). No parsing of the invoice's table structure
happens here or anywhere else in code - the raw text goes straight to the model, which handles
layout variance far more robustly than hand-rolled parsing (see c2f.llm.extract_line_items)."""
from __future__ import annotations

import dataclasses
import pathlib

import pypdf


@dataclasses.dataclass
class Case:
    policy: str
    description: str
    invoice_text: str


def read_case(case_dir: pathlib.Path) -> Case:
    policy = (case_dir / "policy.txt").read_text(encoding="utf-8")
    description = (case_dir / "description.txt").read_text(encoding="utf-8")
    reader = pypdf.PdfReader(str(case_dir / "invoices.pdf"))
    invoice_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return Case(policy=policy, description=description, invoice_text=invoice_text)
