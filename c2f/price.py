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

`other`: a second, independent estimate for the same item (the fast pass, when pricing the
full pass). It is a cheap disagreement check, not a vote:
- coverage split (one says covered, the other doesn't): neither call clears the confidence a
  payout needs, so price as uncertain the same way as "not covered" - b = 0.
- both say covered but t_mid lands far apart (>= DISAGREEMENT_RATIO): use the lower, better-
  supported mid instead of blindly trusting the full pass, and widen the belief by the gap
  between them - this pulls both a and b down, b faster since it already sits below the median.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

UNCOVERED_CHARGE = 0.6  # fraction of the would-be price we charge on uncovered items
B_QUANTILE = 0.27  # c2f.autotune over games 1-17: +1,801 vs 0.20, better in 8 games, worse in 4 (4,399 of it from game 17); backtest 13-17 still 4/5.
# The earlier walk down from 0.27 to 0.18 (games 6-10) overshot slightly: the market's t now runs
# ABOVE our t_mid (calibrated bias 1.19), so refused fair charges cost more than they used to.
RISK_AVERSION = 0.55  # charge maximises mean - RISK_AVERSION * sd of the per-opponent payout.
# c2f.autotune over games 1-14: +8,906 vs 0.585, better in 9 games and worse in 2. It matches the
# post-mortem (UNDERCHARGE 78k, UNDER_ESTIMATE 97k - we were too timid on both sides) and the
# calibrated bias of 1.19, which says the market's t sits ABOVE our t_mid.
#
# 0.30-0.50 scored better still (+10,415) and the backtest passed, but all three break the rail
# this term exists for: below ~0.52 the charge stops FALLING as the belief widens, because a
# slowly-decaying acceptance curve (fitted k = 0.6) makes overcharging look profitable right up to
# the CAP_MULT ceiling. That is true of today's lenient field and false the moment it tightens, and
# crossing t costs ~all the revenue on that line. 0.55 buys 86% of the gain and keeps the rail.
# 0.85 was rejected by the backtest gate rather than the total: only 2/5 recent games profitable.
CAP_MULT = 4.0  # an accepted over-charge pays at most min(a, c) with c >= 4t (the rules' cap)
MODEL_SPREAD_Z = 1.2816  # the model's t_low..t_high is read as an 80 % interval
N_GRID = 200

DISAGREEMENT_RATIO = 1.6  # two passes this far apart on t_mid: use the lower one, don't average
CAP_UNCERTAIN_B_SHRINK = 0.7  # extra caution on b when a referenced policy cap's value is unknown

class InvalidEstimateError(ValueError):
    """A covered+related item reached pricing with t_low=t_mid=t_high<=0 - the SYSTEM
    contract (c2f.llm.SYSTEM) requires a positive t_mid for such items. This must be
    repaired or replaced upstream (c2f.validate + the targeted repair pass in c2f.run),
    never priced here as if the item were uncovered - see the Game 10 item 3 postmortem."""

    def __init__(self, index: object, message: str):
        self.index = index
        super().__init__(message)


BIAS_RANGE = (0.5, 1.5)
SIGMA_RANGE = (0.15, 1.0)
P0_RANGE = (0.02, 0.8)
K_RANGE = (0.0, 6.0)
#: description keyword -> bucket. First match wins, so order matters: the specific
#: trades come before the generic "labour" and "material" catch-alls.
BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("ancillary/call-out", ("vehicle", "travel", "call-out", "callout", "shipping", "delivery",
                            "disposal", "admin", "postage", "mileage")),
    ("diagnostic/inspect", ("diagnos", "inspect", "assess", "report", "survey", "measurement",
                            "moisture", "leak detection", "detection")),
    ("drying/remediation", ("drying", "dehumid", "dryer", "fan", "vacuum", "extract",
                            "stabilis", "stabiliz")),
    ("electronics", ("tv", "television", "speaker", "laptop", "computer", "monitor", "console",
                     "camera", "phone", "audio", "electronic", "appliance")),
    ("valuables", ("watch", "jewel", "ring", "gold", "bicycle", "bike", "sunglasses", "designer")),
    ("labour", ("hour", "hrs", "worker", "technician", "helper", "labour", "labor", "fitter")),
    ("restoration/repair", ("restor", "repair", "renovat", "conservat", "paint", "tiling",
                            "plaster", "skirting", "floor", "wall")),
    ("material", ("material", "parts", "supply", "consumable")),
]


def bucket_of(description: str) -> str:
    d = (description or "").lower()
    for name, keys in BUCKETS:
        if any(k in d for k in keys):
            return name
    return "other"


CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "runs" / "calibration.json"

_N = NormalDist()


@dataclass(frozen=True)
class Calibration:
    bias: float = 1.0  # true t / model t_mid, median
    sigma: float = 0.4  # log-sd of true t around bias * t_mid
    p0: float = 0.35  # fraction of reviewers accepting a charge just over t
    k: float = 2.0  # acceptance decays as (a/t)^-k beyond t
    #: bucket -> bias, where the labels support it. One global bias averages over
    #: categories whose errors point in OPPOSITE directions: measured over games 12-17
    #: we under-price material (median t_lo/t_mid 2.15) and labour (1.52) while
    #: over-pricing drying/remediation (0.74) and restoration/repair (0.72), so a single
    #: multiplier of 1.37 makes those last two worse. c2f.calibrate fits each bucket and
    #: shrinks it toward the global one, so a thin bucket cannot swing far on 5 labels.
    bias_by_bucket: dict[str, float] = field(default_factory=dict)

    def bias_for(self, description: str | None) -> float:
        """Bucket bias when we have one, else the global. Never guesses from a blank."""
        if not description:
            return self.bias
        return self.bias_by_bucket.get(bucket_of(description), self.bias)


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
    raw = d.get("bias_by_bucket") or {}
    buckets = {}
    if isinstance(raw, dict):
        for name, v in raw.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                buckets[str(name)] = _clamp(fv, BIAS_RANGE)
    return Calibration(
        bias=_clamp(vals["bias"], BIAS_RANGE),
        sigma=_clamp(vals["sigma"], SIGMA_RANGE),
        p0=_clamp(vals["p0"], P0_RANGE),
        k=_clamp(vals["k"], K_RANGE),
        bias_by_bucket=buckets,
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
        return cls(mu=math.log(mid * cal.bias_for(est.get("_description"))), sigma=_clamp(max(cal.sigma, model_sigma), SIGMA_RANGE))


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


def price_item(est: dict, cal: Calibration | None = None, other: dict | None = None) -> tuple[float, float]:
    """Return (a, b) as gross totals, both finite and >= 0.

    `other`, if given, is a second independent estimate for the same item - see module
    docstring for how a coverage split or a wide t_mid gap between the two is handled."""
    cal = cal or calibration()
    covered = bool(est.get("covered", False)) and bool(est.get("related", True))
    mid = sorted(_num(est.get(k)) for k in ("t_low", "t_mid", "t_high"))[1]

    other_covered = other_mid = None
    if other is not None:
        other_covered = bool(other.get("covered", False)) and bool(other.get("related", True))
        other_mid = sorted(_num(other.get(k)) for k in ("t_low", "t_mid", "t_high"))[1]

    if other_covered is not None and other_covered != covered:
        # split coverage call: not confident enough either way to pay out
        guess = _num(est.get("t_if_covered")) or mid or other_mid or _num(other.get("t_if_covered"))
        return round(UNCOVERED_CHARGE * guess, 2), 0.0

    if not covered:
        guess = _num(est.get("t_if_covered")) or mid
        return round(UNCOVERED_CHARGE * guess, 2), 0.0

    if mid <= 0:
        raise InvalidEstimateError(
            est.get("index"),
            f"item {est.get('index')}: covered and related but t_low=t_mid=t_high<=0 - "
            "invalid state, must be repaired before pricing, not priced as uncovered",
        )

    est_for_belief, disagreement_sigma = est, 0.0
    if other_mid and mid > 0 and max(mid, other_mid) / min(mid, other_mid) >= DISAGREEMENT_RATIO:
        est_for_belief = {**est, "t_mid": min(mid, other_mid)}
        disagreement_sigma = (math.log(max(mid, other_mid)) - math.log(min(mid, other_mid))) / (2 * MODEL_SPREAD_Z)

    belief = Belief.from_estimate(est_for_belief, cal)
    if disagreement_sigma:
        belief = Belief(mu=belief.mu, sigma=_clamp(max(belief.sigma, disagreement_sigma), SIGMA_RANGE))

    a = best_charge(belief, cal)
    b = accept_limit(belief)
    if bool(est.get("cap_uncertain")):
        b *= CAP_UNCERTAIN_B_SHRINK
    return round(a, 2), round(b, 2)


def price_all(estimates: list[dict], other_output: dict | None = None) -> list[dict]:
    """`other_output`, if given, is a second model's raw {items: [...]} output for the same
    case - used as the disagreement check in price_item (see module docstring)."""
    cal = calibration()
    other_by_idx: dict[int, dict] = {}
    for it in (other_output or {}).get("items", []):
        try:
            other_by_idx[int(it["index"])] = it
        except (KeyError, TypeError, ValueError):
            continue
    out = []
    for est in estimates:
        idx = int(est["index"])
        a, b = price_item(est, cal, other=other_by_idx.get(idx))
        out.append({"index": idx, "charge_price": a, "acceptance_limit": b})
    return out
