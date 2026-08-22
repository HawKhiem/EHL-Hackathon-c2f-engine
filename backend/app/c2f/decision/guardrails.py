"""Last check before the POST.

Runs in well under a millisecond and blocks the submission on anything it cannot
repair. The failure mode this exists to prevent is a silent factor-of-ten: a unit
price submitted where a gross total was required, or a quantity applied twice.

One rule is deliberately **absent**: we do not require `b >= a`. Charging 900
while refusing to pay 900 for the same item is correct here — the optimal charge
tracks the spread of our belief while the acceptance limit is pinned at the 2/3
confidence point, and they move in opposite directions as uncertainty grows. An
invariant that "fixes" it would cost money every round.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from app.c2f.models import (
    ACCEPT_THRESHOLD,
    MAX_CHARGE_MULTIPLE,
    Calibration,
    ItemDecision,
    ItemInference,
    LineItem,
)

#: The charge grid starts here, so anything below it did not come from
#: `choose_a` and is a bug rather than a strategy.
MIN_CHARGE_SHARE: float = 0.02
#: Tolerance on the gross-median derivation. Tight on purpose: this is the check
#: that catches a skipped or doubled quantity multiply.
GROSS_TOLERANCE: float = 0.01
#: Loose bounds on `sum(a)` against an independently stated invoice total.
TOTAL_LOW_MULTIPLE: float = 0.3
TOTAL_HIGH_MULTIPLE: float = 3.0

_SEVERITIES = ("blocking", "warning")


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    detail: str
    item_id: str | None = None
    severity: str = "blocking"

    def __str__(self) -> str:
        where = f" [{self.item_id}]" if self.item_id else ""
        return f"{self.severity}: {self.rule}{where} - {self.detail}"


def check(
    items: Sequence[LineItem],
    decisions: Sequence[ItemDecision],
    *,
    inferences: Sequence[ItemInference] | None = None,
    calibration: Calibration | None = None,
    invoice_total: float | None = None,
) -> list[Violation]:
    """Every rule from `docs/DESIGN.md` section 9. Empty list means submit."""
    violations: list[Violation] = []
    by_id = {decision.item_id: decision for decision in decisions}
    quantities = {i.item_id: i.quantity for i in items}
    inference_by_id = {i.item_id: i for i in (inferences or ())}
    calibration = calibration or Calibration()

    # 6. ids and count line up with the parsed invoice
    item_ids = [item.item_id for item in items]
    missing = [i for i in item_ids if i not in by_id]
    extra = [d.item_id for d in decisions if d.item_id not in set(item_ids)]
    if missing:
        violations.append(Violation("item_coverage", f"no decision for {missing}"))
    if extra:
        violations.append(Violation("item_coverage", f"decision for unknown items {extra}"))
    if len(decisions) != len(set(d.item_id for d in decisions)):
        violations.append(Violation("item_coverage", "duplicate item_id in decisions"))

    for decision in decisions:
        ceiling = decision.q50_gross * MAX_CHARGE_MULTIPLE

        # 1. present and numeric
        for name, value in (("a", decision.a), ("b", decision.b)):
            if value is None or not math.isfinite(value):
                violations.append(Violation("finite", f"{name}={value!r}", decision.item_id))

        if not math.isfinite(decision.a) or not math.isfinite(decision.b):
            continue

        # 2. inside the plausible band
        if decision.a < 0 or decision.a > ceiling:
            violations.append(
                Violation(
                    "charge_bounds",
                    f"a={decision.a:.2f} outside [0, {ceiling:.2f}]",
                    decision.item_id,
                )
            )
        if decision.b < 0 or decision.b > ceiling:
            violations.append(
                Violation(
                    "limit_bounds",
                    f"b={decision.b:.2f} outside [0, {ceiling:.2f}]",
                    decision.item_id,
                )
            )

        # 3a. gross total, not a unit price. Checked on the *derivation* rather
        # than the magnitude: at quantity 2 a skipped multiply is indistinguishable
        # from a legitimately low charge, but `quantity x unit_q50` never is.
        source = inference_by_id.get(decision.item_id)
        if source is not None:
            quantity = quantities.get(decision.item_id, decision.quantity) or 1.0
            shrink = min(max(source.skeptic_multiplier, 0.5), 1.0)
            expected = source.unit_quantiles.q50 * quantity * shrink * calibration.mu_shift
            if expected > 0 and abs(decision.q50_gross / expected - 1.0) > GROSS_TOLERANCE:
                violations.append(
                    Violation(
                        "gross_total",
                        f"q50_gross={decision.q50_gross:.2f} but quantity x unit q50 "
                        f"x skeptic x mu_shift = {expected:.2f} - quantity applied "
                        f"{decision.q50_gross / expected:.2f}x",
                        decision.item_id,
                    )
                )

        # 3b. cheap magnitude backstop for items with no inference to check against
        floor = decision.q50_gross * MIN_CHARGE_SHARE
        if 0 < decision.a < floor:
            violations.append(
                Violation(
                    "charge_floor",
                    f"a={decision.a:.2f} is under {MIN_CHARGE_SHARE:.0%} of the gross "
                    f"median {decision.q50_gross:.2f}, below the charge grid",
                    decision.item_id,
                )
            )

        # 4. the 2/3 bar is exact
        if decision.p_valid < ACCEPT_THRESHOLD and decision.b != 0.0:
            violations.append(
                Violation(
                    "accept_threshold",
                    f"p_valid={decision.p_valid:.3f} below 2/3 but b={decision.b:.2f}",
                    decision.item_id,
                )
            )

        # 5. never leave the free option on the table
        if decision.a <= 0.0:
            violations.append(
                Violation(
                    "zero_charge",
                    "a=0 collects nothing and a rejected charge costs the issuer nothing",
                    decision.item_id,
                    severity="warning",
                )
            )

    # 7. sanity against an independently stated invoice total
    if invoice_total and invoice_total > 0:
        charged = sum(d.a for d in decisions if math.isfinite(d.a))
        low, high = invoice_total * TOTAL_LOW_MULTIPLE, invoice_total * TOTAL_HIGH_MULTIPLE
        if not low <= charged <= high:
            violations.append(
                Violation(
                    "invoice_total",
                    f"sum(a)={charged:.2f} outside [{low:.2f}, {high:.2f}] "
                    f"for stated total {invoice_total:.2f}",
                    severity="warning",
                )
            )

    return violations


def repair(decisions: Sequence[ItemDecision]) -> list[ItemDecision]:
    """Clamp what can be clamped so a single bad item cannot lose the round.

    Only touches values, never the set of items — a missing item is a parsing
    failure and has to be handled upstream.
    """
    fixed: list[ItemDecision] = []
    for decision in decisions:
        ceiling = decision.q50_gross * MAX_CHARGE_MULTIPLE
        notes = list(decision.notes)

        a = decision.a if math.isfinite(decision.a) else decision.q50_gross
        b = decision.b if math.isfinite(decision.b) else 0.0
        a = min(max(a, 0.0), ceiling)
        b = min(max(b, 0.0), ceiling)

        if decision.p_valid < ACCEPT_THRESHOLD and b != 0.0:
            b = 0.0
            notes.append("repaired: b forced to 0 by the 2/3 bar")
        if a <= 0.0 < decision.q50_gross:
            a = decision.q50_gross
            notes.append("repaired: a raised off zero to the free-option median")

        if (a, b) != (decision.a, decision.b):
            fixed.append(replace(decision, a=a, b=b, notes=notes))
        else:
            fixed.append(decision)
    return fixed


def blocking(violations: Iterable[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == "blocking"]
