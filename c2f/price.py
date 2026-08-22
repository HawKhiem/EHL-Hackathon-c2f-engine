"""Turn the model's per-item estimate into (a, b). Pure math, no I/O except the calibration file.

a = charge price (what we bill every other team)
b = acceptance limit (max we pay when billed the same item)

Belief on the fair value t: lognormal, median = t_mid * bias, sigma = max(calibrated sigma,
the model's own spread). `bias` and `sigma` are learned by c2f.calibrate from the t bounds
c2f.truth recovers after each game (runs/calibration.json); defaults apply until then.

- a maximises risk-adjusted revenue, mean - RISK_AVERSION * sd per opponent. A fair charge
  (a <= t) is paid by every opponent; a fraudulent one only by the reviewers whose limit is
  still above it, a fraction that decays with how far over t it is: p_accept(a/t) = p0 *
  (a/t)^-k (also learned from the market). The sd term matters: pure expectation chases the
  belief's upper tail and charges MORE the less sure we are; with it, the charge moves from
  ~0.7x the median (sigma 0.2) to ~0.5x (sigma 0.8) - crossing t loses ~all the revenue.
- b = 1/3-quantile of the belief. Accepting a fair charge costs 1x, wrongly rejecting 1.5x,
  accepting fraud 1x -> accept iff P(t >= a') > 2/3.
- not covered / not related: b = 0, a = UNCOVERED_CHARGE * t_if_covered (0 if unknown):
  a rejected fraudulent charge costs the issuer nothing, so this is free upside.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

UNCOVERED_CHARGE = 0.6  # fraction of the would-be price we charge on uncovered items
B_QUANTILE = 0.27
RISK_AVERSION = 0.585  # charge maximises mean - RISK_AVERSION * sd of the per-opponent payout
CAP_MULT = 4.0  # an accepted over-charge pays at most min(a, c) with c >= 4t (the rules' cap)
MODEL_SPREAD_Z = 1.2816  # the model's t_low..t_high is read as an 80 % interval
N_GRID = 200

BIAS_RANGE = (0.5, 1.5)
SIGMA_RANGE = (0.15, 1.0)
P0_RANGE = (0.02, 0.8)
K_RANGE = (0.0, 6.0)
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "runs" / "calibration.json"

_N = NormalDist()


@dataclass(frozen=True)
class Calibration:
    bias: float = 1.0  # true t / model t_mid, median
    sigma: float = 0.4  # log-sd of true t around bias * t_mid
    p0: float = 0.35  # fraction of reviewers accepting a charge just over t
    k: float = 2.0  # acceptance decays as (a/t)^-k beyond t


DEFAULT_CALIBRATION = Calibration()


def _clamp(v: float, rng: tuple[float, float]) -> float:
    return min(rng[1], max(rng[0], v))


def calibration() -> Calibration:
    """Learned calibration (runs/calibration.json), clamped; the default if missing or broken."""
    try:
        d = json.loads(CALIBRATION_PATH.read_text())
        vals = {k: float(d[k]) for k in ("bias", "sigma", "p0", "k")}
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_CALIBRATION
    if not all(math.isfinite(v) for v in vals.values()):
        return DEFAULT_CALIBRATION
    return Calibration(
        bias=_clamp(vals["bias"], BIAS_RANGE),
        sigma=_clamp(vals["sigma"], SIGMA_RANGE),
        p0=_clamp(vals["p0"], P0_RANGE),
        k=_clamp(vals["k"], K_RANGE),
    )


def _num(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


@dataclass(frozen=True)
class Belief:
    """Lognormal belief on t: ln t ~ N(mu, sigma^2)."""

    mu: float
    sigma: float

    @property
    def median(self) -> float:
        return math.exp(self.mu)

    def quantile(self, q: float) -> float:
        return math.exp(self.mu + self.sigma * _N.inv_cdf(q))

    @classmethod
    def from_estimate(cls, est: dict, cal: Calibration) -> "Belief":
        lo, mid, hi = sorted(_num(est.get(k)) for k in ("t_low", "t_mid", "t_high"))
        model_sigma = (math.log(hi) - math.log(lo)) / (2 * MODEL_SPREAD_Z) if lo > 0 and hi > lo else 0.0
        return cls(mu=math.log(mid * cal.bias), sigma=_clamp(max(cal.sigma, model_sigma), SIGMA_RANGE))


def accept_limit(belief: Belief) -> float:
    return belief.quantile(B_QUANTILE)


def best_charge(belief: Belief, cal: Calibration, n_grid: int = N_GRID, risk_aversion: float | None = None) -> float:
    """Charge maximising mean - risk_aversion * sd of the per-opponent payout, over the belief's quantiles.

    Never above the median (a charge more likely fraud than fair is not a price, it is a bet), and an
    accepted over-charge pays at most CAP_MULT * t - without that cap a slowly decaying acceptance
    curve (k < 1) would reward charging the moon."""
    if risk_aversion is None:
        risk_aversion = RISK_AVERSION
    ts = [belief.quantile((j + 0.5) / n_grid) for j in range(n_grid)]
    best_a, best_v = 0.0, -math.inf
    for a in ts:
        if a > belief.median:
            break
        pays = [a if a <= t else min(a, CAP_MULT * t) * cal.p0 * (t / a) ** cal.k for t in ts]
        mean = sum(pays) / n_grid
        sd = math.sqrt(sum((x - mean) ** 2 for x in pays) / n_grid)
        v = mean - risk_aversion * sd
        if v > best_v:
            best_a, best_v = a, v
    return best_a


def price_item(est: dict, cal: Calibration | None = None) -> tuple[float, float]:
    """Return (a, b) as gross totals, both finite and >= 0."""
    cal = cal or calibration()
    covered = bool(est.get("covered", False)) and bool(est.get("related", True))
    mid = sorted(_num(est.get(k)) for k in ("t_low", "t_mid", "t_high"))[1]
    if not covered or mid <= 0:
        guess = _num(est.get("t_if_covered")) or mid
        return round(UNCOVERED_CHARGE * guess, 2), 0.0
    belief = Belief.from_estimate(est, cal)
    return round(best_charge(belief, cal), 2), round(accept_limit(belief), 2)


def price_all(estimates: list[dict]) -> list[dict]:
    cal = calibration()
    out = []
    for est in estimates:
        a, b = price_item(est, cal)
        out.append({"index": int(est["index"]), "charge_price": a, "acceptance_limit": b})
    return out
