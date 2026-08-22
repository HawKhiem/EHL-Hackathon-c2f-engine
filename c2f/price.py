"""Turn the model's per-item estimate into (a, b). Pure math, no I/O.

a = charge price (what we bill every other team)
b = acceptance limit (max we pay when billed the same item)

Rules (see docs/superpowers/specs/2026-08-22-c2f-engine-design.md):
- b = 1/3-quantile of a triangular(t_low, t_mid, t_high) belief on t.
  Accepting a fair charge costs 1x, wrongly rejecting costs 1.5x, accepting fraud
  costs 1x -> accept iff P(t >= a') > 2/3.
- a = t_mid * (1 - K_UNCERTAINTY * spread), never below t_low.
- b is computed on the belief scaled by B_SCALE (>= 1). Games 1-4 showed the model's t
  estimates sit BELOW the true t (every rejection in the market so far was of a fair
  charge), so the accept limit is pushed up. `c2f.calibrate` learns the scale from
  `c2f.truth` bounds and writes runs/calibration.json; the default applies until then.
- not covered / not related: b = 0, a = UNCOVERED_CHARGE * t_if_covered (0 if unknown).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

K_UNCERTAINTY = 0.25  # how much of the relative spread we shave off a
UNCOVERED_CHARGE = 0.6  # fraction of the would-be price we charge on uncovered items
B_QUANTILE = 1 / 3
B_SCALE_DEFAULT = 1.3  # belief scale for b when no calibration file exists
B_SCALE_RANGE = (1.0, 2.0)
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "runs" / "calibration.json"


def b_scale() -> float:
    """Learned belief scale for the accept limit (runs/calibration.json), else the default."""
    try:
        v = float(json.loads(CALIBRATION_PATH.read_text())["b_scale"])
    except (OSError, ValueError, KeyError, TypeError):
        return B_SCALE_DEFAULT
    lo, hi = B_SCALE_RANGE
    return min(hi, max(lo, v)) if math.isfinite(v) else B_SCALE_DEFAULT


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


def price_item(est: dict, scale: float | None = None) -> tuple[float, float]:
    """Return (a, b) as gross totals, both finite and >= 0. `scale` overrides the b belief scale."""
    covered = bool(est.get("covered", False)) and bool(est.get("related", True))
    lo, mid, hi = (_num(est.get(k)) for k in ("t_low", "t_mid", "t_high"))
    lo, mid, hi = sorted((lo, mid, hi))

    if not covered or mid <= 0:
        guess = _num(est.get("t_if_covered")) or mid
        return round(UNCOVERED_CHARGE * guess, 2), 0.0

    spread = (hi - lo) / mid if mid else 0.0
    a = max(lo, mid * (1 - K_UNCERTAINTY * spread))
    s = b_scale() if scale is None else scale
    b = tri_quantile(lo * s, mid * s, hi * s, B_QUANTILE)
    return round(a, 2), round(b, 2)


def price_all(estimates: list[dict]) -> list[dict]:
    s = b_scale()
    out = []
    for est in estimates:
        a, b = price_item(est, scale=s)
        out.append({"index": int(est["index"]), "charge_price": a, "acceptance_limit": b})
    return out
