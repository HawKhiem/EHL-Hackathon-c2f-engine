"""Prompts for the semantic layer.

Design rules these encode:

* One call sees the **whole invoice** and returns an array. Per-item calls do not
  fit in 60 seconds, and an item-at-a-time view cannot spot a duplicated line or
  a labour charge inconsistent with the parts.
* `p_valid` is asked for **jointly**. `p_covered * p_related` understates it
  because the two are strongly dependent.
* Prices are asked for **per unit**. The quantity multiply happens in code.
* No prompt is told about `a`, `b`, the 2/3 rule, or the payoff table. The
  semantic layer answers "what do the documents imply"; the decision is not its
  business, and telling it about the incentives only invites it to skew the
  estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.c2f.models import LineItem

VALIDITY_SYSTEM = """\
You are a senior insurance claims examiner. You are given an insurance policy, a \
damage report, and the line items of a repair invoice.

For each line item, estimate the probability that it is BOTH covered by the policy \
AND reasonably necessary as a consequence of the reported damage. Estimate that \
joint probability directly - do not estimate two separate probabilities and \
multiply them.

Calibration matters more than caution. If an item is plainly part of the described \
repair and plainly inside the policy, say 0.97, not 0.8. If the policy explicitly \
excludes it, say 0.02. Reserve middling values for genuine ambiguity.

Return ONLY a JSON array, one object per line item, no prose:
[
  {
    "item_id": "<exactly the id given to you>",
    "p_valid": 0.0,
    "p_covered": 0.0,
    "p_related": 0.0,
    "evidence": "<one sentence, quoting the policy clause or damage detail>",
    "exclusion_hit": "<policy exclusion that applies, or empty>"
  }
]
"""

PRICING_SYSTEM = """\
You are a German repair-cost assessor (Sachverständiger) pricing trade work.

For each invoice line item, give the plausible range of a FAIR market price. \
Assume the item is legitimate and necessary - somebody else judges that.

Rules:
- Price ONE UNIT of the item, not the whole line. Ignore the quantity entirely.
- For labour lines, one unit is one hour (or whatever unit is stated).
- Include applicable VAT in the figures.
- Prices are in EUR.
- q10 is a cheap-but-real price, q50 a typical workshop price, q90 an expensive \
but still defensible one. Make the spread reflect your actual uncertainty: narrow \
for a commodity part you know well, wide for a vague description.

Return ONLY a JSON array, one object per line item, no prose:
[
  {
    "item_id": "<exactly the id given to you>",
    "q10": 0, "q25": 0, "q50": 0, "q75": 0, "q90": 0,
    "unit_basis": "<what one unit is>",
    "price_basis": "<one sentence on where the number comes from>",
    "confidence": "high|medium|low"
  }
]
"""

SKEPTIC_SYSTEM = """\
You are an adversarial claims auditor. Assume the invoice may be padded and look \
for it: work that is not needed for the described damage, quantities that exceed \
what the job requires, replacement where repair would do, premium parts on a \
standard repair, duplicated lines, labour hours inconsistent with the parts fitted.

You are looking across the WHOLE invoice, so use cross-item evidence.

For each line item return a multiplier on the fair unit price - 1.0 if you find \
nothing wrong, down to 0.5 if the line looks substantially padded. Do not exceed \
1.0; you argue prices down, never up.

Return ONLY a JSON array, one object per line item, no prose:
[
  {
    "item_id": "<exactly the id given to you>",
    "suggested_multiplier": 1.0,
    "inflation_flags": ["..."],
    "qty_plausible": true
  }
]
"""


def render_items(items: Sequence[LineItem]) -> str:
    """The invoice as the models see it. Ids are ours, and they must come back."""
    lines = []
    for item in items:
        unit = f" {item.unit}" if item.unit else ""
        lines.append(
            f"- item_id={item.item_id} | quantity={item.quantity:g}{unit} | {item.description}"
        )
    return "\n".join(lines)


def case_context(
    items: Sequence[LineItem],
    *,
    policy: str = "",
    description: str = "",
    include_policy: bool = True,
) -> str:
    """One user message carrying the whole case.

    `include_policy` is off for the pricing call: the policy says nothing about
    market rates, and dropping it keeps that prompt short and fast.
    """
    blocks: list[str] = []
    if include_policy and policy.strip():
        blocks.append(f"=== INSURANCE POLICY ===\n{policy.strip()}")
    if description.strip():
        blocks.append(f"=== DAMAGE DESCRIPTION ===\n{description.strip()}")
    blocks.append(f"=== INVOICE LINE ITEMS ({len(items)}) ===\n{render_items(items)}")
    blocks.append(f"Return exactly {len(items)} objects, one per item_id above, in the same order.")
    return "\n\n".join(blocks)
