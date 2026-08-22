"""The decision layer: `(a, b)` from a survival function.

Pure functions of `(S, G, cap)`. No LLM, no I/O, no clock. That is what lets us
replay every historical round through a changed policy in milliseconds, which is
the only realistic way to learn anything inside 100 rounds.

Two results from the payoff table drive the whole module (`docs/DESIGN.md`):

* When `a <= t` the issuer is paid whether the insurer accepts or rejects, so the
  honest part of charge revenue does not depend on any opponent model at all.
* `p_valid` scales `S` uniformly, so it cancels out of `argmax a*S(a)` — it moves
  `b` and leaves `a` alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from app.c2f.models import (
    ACCEPT_THRESHOLD,
    MAX_CHARGE_MULTIPLE,
    Calibration,
    ItemDecision,
    ItemInference,
    LineItem,
)
from app.c2f.probability.survival import MIN_PRICE, FairValueDistribution

#: Probability that an average opponent accepts a charge of `a`.
AcceptanceModel = Callable[[float], float]

_GRID_POINTS: int = 2000
_GRID_LOW_MULTIPLE: float = 0.02
#: Prefer the cheapest charge within this much of the best expected revenue.
#: Flat optima are common; the low end is the one that survives our own
#: median being too optimistic.
_TIE_TOLERANCE: float = 1e-3


def charge_grid(
    dist: FairValueDistribution,
    *,
    cap: float | None = None,
    points: int = _GRID_POINTS,
) -> list[float]:
    """Log-spaced candidate charges.

    Upper bound is `4 x median`: the accepted-payment cap satisfies `c >= 4t`, so
    anything above that is waste even in the best case.
    """
    low = max(dist.median * _GRID_LOW_MULTIPLE, MIN_PRICE)
    high = dist.median * MAX_CHARGE_MULTIPLE
    if cap is not None and cap > low:
        high = min(high, cap)
    if high <= low:
        return [low]

    log_low, log_high = math.log(low), math.log(high)
    step = (log_high - log_low) / (points - 1)
    return [math.exp(log_low + step * i) for i in range(points)]


def expected_revenue(
    price: float,
    dist: FairValueDistribution,
    *,
    acceptance: AcceptanceModel | None = None,
    cap: float | None = None,
) -> float:
    """`a*S(a) + min(a,c)*(1-S(a))*G(a)`, per opponent.

    First term: the charge is fair, so we are owed it regardless of the
    opponent's limit. Second term: the charge is inflated, so we only collect if
    that particular opponent accepts. Nothing here can be negative — as issuer
    the worst case is being paid nothing.
    """
    survival = dist.survival(price)
    revenue = price * survival
    if acceptance is not None:
        paid = min(price, cap) if cap is not None else price
        revenue += paid * (1.0 - survival) * acceptance(price)
    return revenue


def choose_a(
    dist: FairValueDistribution,
    *,
    acceptance: AcceptanceModel | None = None,
    cap: float | None = None,
    lambda_a: float = 1.0,
    points: int = _GRID_POINTS,
) -> float:
    """Revenue-maximising charge.

    With no acceptance model this is `argmax a*S(a)`, which is a competitive
    strategy on its own and needs zero opponent data. Note it rises with our
    uncertainty: at log-spread 0.4 the optimum sits near the 24th percentile of
    our belief, at 0.8 near the median, at 1.2 near the 72nd.
    """
    best_price: float | None = None
    best_revenue = 0.0
    for price in charge_grid(dist, cap=cap, points=points):
        revenue = expected_revenue(price, dist, acceptance=acceptance, cap=cap)
        if best_price is None or revenue > best_revenue * (1.0 + _TIE_TOLERANCE):
            best_revenue, best_price = revenue, price

    if best_revenue <= 0.0:
        # Our belief says nothing here is collectable. Charging anyway is free:
        # a rejected issuer pays no penalty, it just collects nothing. So take
        # the free option at our median rather than submitting a=0 and
        # guaranteeing zero.
        best_price = dist.median

    scaled = (best_price or dist.median) * max(lambda_a, 0.0)
    ceiling = dist.median * MAX_CHARGE_MULTIPLE
    if cap is not None:
        ceiling = min(ceiling, cap)
    return min(max(scaled, 0.0), ceiling)


def choose_b(dist: FairValueDistribution, *, lambda_b: float = 1.0) -> float:
    """Highest price we will pay: `sup{a : S(a) >= 2/3}`.

    Solved exactly rather than by grid search. `S(a) >= 2/3` rearranges to
    `F_+(a) <= 1 - 2/(3*p_valid)`, so:

    * `p_valid = 1`   -> the 33rd percentile of the price belief
    * `p_valid = 0.8` -> the 17th percentile
    * `p_valid < 2/3` -> exactly zero, because no positive price can clear the
      bar. We would rather pay the lawyer than fund an item we do not believe in.

    `lambda_b` corrects our price belief, never the 2/3 — that constant comes
    straight from the payoff table and is not ours to tune.
    """
    if dist.p_valid < ACCEPT_THRESHOLD:
        return 0.0
    level = 1.0 - ACCEPT_THRESHOLD / dist.p_valid
    return max(dist.quantile_positive(level) * max(lambda_b, 0.0), 0.0)


def decide(
    item: LineItem,
    inference: ItemInference,
    *,
    calibration: Calibration | None = None,
    acceptance: AcceptanceModel | None = None,
    cap: float | None = None,
) -> ItemDecision:
    """Full decision for one line item, with its own audit trail."""
    calibration = calibration or Calibration()
    dist = FairValueDistribution.from_inference(
        inference, quantity=item.quantity, calibration=calibration
    )

    a = choose_a(dist, acceptance=acceptance, cap=cap, lambda_a=calibration.lambda_a)
    b = choose_b(dist, lambda_b=calibration.lambda_b)

    notes: list[str] = []
    if inference.degraded:
        notes.append("inference degraded: heuristic price belief")
    if b == 0.0:
        notes.append(f"b=0: p_valid {inference.p_valid:.2f} below the 2/3 acceptance bar")
    if b < a:
        notes.append("b<a is expected here, not a bug: see docs/DESIGN.md R2")

    return ItemDecision(
        item_id=item.item_id,
        a=a,
        b=b,
        s_at_a=dist.survival(a),
        s_at_b=dist.survival(b),
        sigma_log=dist.sigma_log,
        q50_gross=dist.median,
        p_valid=dist.p_valid,
        quantity=item.quantity,
        reason=(
            f"median {dist.median:.2f}, log-spread {dist.sigma_log:.2f}, p_valid {dist.p_valid:.2f}"
        ),
        notes=notes,
    )
