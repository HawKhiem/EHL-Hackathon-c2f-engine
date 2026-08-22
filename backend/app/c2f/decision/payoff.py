"""The payoff table, executable.

Everything else in the system is an estimate. This module is the ground truth:
given a charge, an acceptance limit, the secret threshold and the cap, it says
exactly what money moves. Used to score a strategy offline against a known `t`,
and later to run the Monte Carlo that picks `b` directly.

Validated against the worked example in `GAME_DESCRIPTION.md` - if that test
fails, this module is wrong, not the test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: The insurer's penalty multiplier for rejecting a fair charge: it still owes
#: `a`, plus `0.5a` to the lawyer.
WRONGFUL_REJECT_MULTIPLE: float = 1.5
#: `c >= 4t`. Using the lower bound makes any simulation pessimistic about
#: overcharging, which is the safe direction to be wrong in.
MIN_CAP_MULTIPLE: float = 4.0


@dataclass(frozen=True, slots=True)
class Transaction:
    """One issuer-insurer pair on one line item."""

    issuer_receives: float
    insurer_pays: float
    accepted: bool
    fair: bool

    @property
    def label(self) -> str:
        if self.fair and self.accepted:
            return "fair, accepted"
        if self.fair:
            return "wrongful reject"
        if self.accepted:
            return "fraud, accepted"
        return "rightful reject"


def transaction(
    charge: float, limit: float, threshold: float, *, cap: float | None = None
) -> Transaction:
    """Resolve one transaction exactly as the payoff table specifies.

    Note the asymmetry that shapes the whole strategy: in the fair zone the
    issuer is paid whether or not the insurer accepts, so `limit` only changes
    who suffers, never whether the issuer collects.
    """
    fair = charge <= threshold
    accepted = charge <= limit
    ceiling = cap if cap is not None else threshold * MIN_CAP_MULTIPLE

    if accepted:
        paid = charge if fair else min(charge, ceiling)
        return Transaction(paid, paid, accepted=True, fair=fair)
    if fair:
        return Transaction(charge, charge * WRONGFUL_REJECT_MULTIPLE, accepted=False, fair=True)
    return Transaction(0.0, 0.0, accepted=False, fair=False)


@dataclass(frozen=True, slots=True)
class ItemScore:
    received: float
    paid: float

    @property
    def net(self) -> float:
        return self.received - self.paid


def score_item(
    our_a: float,
    our_b: float,
    opponents: Sequence[tuple[float, float]],
    threshold: float,
    *,
    cap: float | None = None,
) -> ItemScore:
    """Our net on one line item against a field of `(a, b)` opponents.

    Every team is matched against every other in both roles, so each opponent
    contributes twice: once with us issuing, once with us insuring.
    """
    received = 0.0
    paid = 0.0
    for opponent_a, opponent_b in opponents:
        received += transaction(our_a, opponent_b, threshold, cap=cap).issuer_receives
        paid += transaction(opponent_a, our_b, threshold, cap=cap).insurer_pays
    return ItemScore(received, paid)


def score_field(
    submissions: dict[str, tuple[float, float]],
    threshold: float,
    *,
    cap: float | None = None,
) -> dict[str, ItemScore]:
    """Score every team in a field against each other on one line item.

    Reproduces a whole round, which is what makes the brief's worked example a
    usable regression test.
    """
    out: dict[str, ItemScore] = {}
    for team, (our_a, our_b) in submissions.items():
        others = [pair for name, pair in submissions.items() if name != team]
        out[team] = score_item(our_a, our_b, others, threshold, cap=cap)
    return out
