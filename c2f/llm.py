"""OpenAI calls, one enforced-JSON-schema call per concern - never hand-rolled text parsing.
PDF layouts vary too much (wrapped descriptions, skipped indices, glued units) for that to be
reliable, so every extraction step - even pulling {index: description} out of raw invoice
text - is delegated to the model and constrained only by the response schema:

- extract_line_items: {index, description} from the raw invoice text alone. Used both live and
  to build the historical dataset (c2f/history.py).
- estimate_coverage: is this line item covered, given THIS case's policy + damage description?
  Coverage is case-specific and doesn't transfer across games.
- estimate_prices: what's the fair market value of this line item? A pure market-value question
  that should only ever see the invoice (plus historical brackets for similar past items) -
  feeding it the policy/description would bias it with information that has nothing to do with
  price.

No image/vision input on the live path.
"""
from __future__ import annotations

import json

from openai import OpenAI

from c2f import env
from c2f.extract import Case

_ITEMS_SCHEMA = {
    "name": "line_items",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["index", "description"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}

_COVERAGE_SCHEMA = {
    "name": "line_item_coverage",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "description": {"type": "string"},
                        "covered": {"type": "boolean"},
                    },
                    "required": ["index", "description", "covered"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}

_PRICE_SCHEMA = {
    "name": "line_item_prices",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "description": {"type": "string"},
                        "t_low": {"type": "number"},
                        "t_high": {"type": ["number", "null"]},
                    },
                    "required": ["index", "description", "t_low", "t_high"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}

_ITEMS_SYSTEM_PROMPT = """\
Extract every line item from this invoice's raw text. For each, give its position number
("POS.") and its full description, with any wrapped continuation lines merged back into one
description. Ignore the quantity/unit/amount/total columns. List every position number found,
once each."""

_COVERAGE_SYSTEM_PROMPT = """\
You are assessing insurance coverage. For every line item on the invoice, decide whether the
policy covers it given the damage description - it may fail on relevance to the reported
damage as well as on the policy's own exclusions. List every position number ("POS.") found
in the invoice, once each."""

_PRICE_SYSTEM_PROMPT = """\
You are estimating fair market prices for repair/replacement line items, independent of any
insurance policy or coverage question. For every line item on the invoice, give your best
bracket [t_low, t_high] for the fair market GROSS TOTAL price of that whole line item
(quantity * unit price, not a per-unit price), based on typical costs for such work or goods
and the historical reference data provided. t_high may be null if you have no reasonable upper
bound. List every position number ("POS.") found in the invoice, once each."""


def _history_block(history: list[dict]) -> str:
    if not history:
        return "(no historical data available yet)"
    lines = []
    for row in history:
        desc = row["description"] or "(description not recovered)"
        if row["t_hi"] is None:
            bracket = f">= EUR {row['t_lo']:.0f}"
        else:
            bracket = f"EUR {row['t_lo']:.0f}-{row['t_hi']:.0f}"
        lines.append(f"- {desc}: {bracket}")
    return "\n".join(lines)


_TIMEOUT_S = 45  # each round only allows 60s to submit - a hung call must fail loud, not hang


def _call(system_prompt: str, user_prompt: str, schema: dict) -> list[dict]:
    client = OpenAI(api_key=env.OPENAI_API_KEY, timeout=_TIMEOUT_S, max_retries=1)
    response = client.chat.completions.create(
        model=env.MODEL,
        reasoning_effort=env.REASONING,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    payload = json.loads(response.choices[0].message.content)
    return payload["items"]


def extract_line_items(invoice_text: str) -> list[dict]:
    user_prompt = f"INVOICE (raw text extracted from the PDF):\n{invoice_text}"
    return _call(_ITEMS_SYSTEM_PROMPT, user_prompt, _ITEMS_SCHEMA)


def estimate_coverage(case: Case) -> list[dict]:
    user_prompt = f"""\
POLICY:
{case.policy}

DAMAGE DESCRIPTION:
{case.description}

INVOICE (raw text extracted from the PDF):
{case.invoice_text}
"""
    return _call(_COVERAGE_SYSTEM_PROMPT, user_prompt, _COVERAGE_SCHEMA)


def estimate_prices(invoice_text: str, history: list[dict]) -> list[dict]:
    user_prompt = f"""\
INVOICE (raw text extracted from the PDF):
{invoice_text}

HISTORICAL REFERENCE - real revealed fair-value brackets for similar items from past cases:
{_history_block(history)}
"""
    return _call(_PRICE_SYSTEM_PROMPT, user_prompt, _PRICE_SCHEMA)
