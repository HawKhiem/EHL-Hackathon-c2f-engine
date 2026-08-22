"""Turn the model's per-item estimate into (a, b). Pure math, no I/O.

a = charge price (what we bill every other team)
b = acceptance limit (max we pay when billed the same item)

Rules (see docs/superpowers/specs/2026-08-22-c2f-engine-design.md):
- b = 1/3-quantile of a triangular(t_low, t_mid, t_high) belief on t.
  Accepting a fair charge costs 1x, wrongly rejecting costs 1.5x, accepting fraud
  costs 1x -> accept iff P(t >= a') > 2/3.
- a = t_mid * (1 - K_UNCERTAINTY * spread), never below t_low.
- not covered / not related: b = 0, a = UNCOVERED_CHARGE * t_if_covered (0 if unknown).
"""

from __future__ import annotations

import math

K_UNCERTAINTY = 0.5  # how much of the relative spread we shave off a (game 2 showed ~20% overestimates)
UNCOVERED_CHARGE = 0.6  # fraction of the would-be price we charge on uncovered items
B_QUANTILE = 1 / 3


def tri_quantile(lo: float, mode: float, hi: float, q: float) -> float:
    """Quantile q of a triangular distribution on [lo, hi] with the given mode."""
    lo, mode, hi = sorted((lo, mode, hi))
    if hi == lo:
        return lo
    f_mode = (mode - lo) / (hi - lo)
    if q <= f_mode:
        return lo + math.sqrt(q * (hi - lo) * (mode - lo))
    return hi - math.sqrt((1 - q) * (hi - lo) * (hi - mode))


def _num(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def price_item(est: dict) -> tuple[float, float]:
    """Return (a, b) as gross totals, both finite and >= 0."""
    covered = bool(est.get("covered", False)) and bool(est.get("related", True))
    lo, mid, hi = (_num(est.get(k)) for k in ("t_low", "t_mid", "t_high"))
    lo, mid, hi = sorted((lo, mid, hi))

    if not covered or mid <= 0:
        guess = _num(est.get("t_if_covered")) or mid
        return round(UNCOVERED_CHARGE * guess, 2), 0.0

    spread = (hi - lo) / mid if mid else 0.0
    a = max(lo, mid * (1 - K_UNCERTAINTY * spread))
    b = tri_quantile(lo, mid, hi, B_QUANTILE)
    return round(a, 2), round(b, 2)


def price_all(estimates: list[dict]) -> list[dict]:
    out = []
    for est in estimates:
        a, b = price_item(est)
        out.append({"index": int(est["index"]), "charge_price": a, "acceptance_limit": b})
    return out
