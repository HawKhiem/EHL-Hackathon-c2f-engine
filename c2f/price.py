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
- not covered / not related: b = 0, a = UNCOVERED_CHARGE * bias * t_if_covered (0 if unknown):
  a rejected fraudulent charge costs the issuer nothing, so this is free upside. The same
  bucket `bias` the covered path applies to t_mid applies here: t_if_covered comes out of the
  same call with the same systematic underestimate, so correcting one and not the other was
  an inconsistency, not a choice.

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

UNCOVERED_CHARGE = 0.9  # fraction of the would-be price we charge on uncovered items,
# multiplied by the same bucket bias the covered path uses (1.48 global today, so ~1.34x
# t_if_covered).
# Raised from 0.6: over 19 labelled games (c2f.accuracy) 31 of the 101 items we zeroed were
# proven covered by the market, against only 5 of 136 priced items proven worthless. Wrongly
# rejected fraud costs the issuer nothing, so the only cost of charging near the full
# would-be price on an uncovered item is zero - and the upside is the third of them we
# excluded by mistake.
#
# Splitting the branch over games 18-26 (c2f.deviation): the 8 items the market agreed were
# worthless cost and earn nothing at ANY multiplier, so all of the money is the 13 items we
# zeroed that the market proved had value (game 20 item 1 alone is 61% of the regret).
# Expected revenue on those rises to a peak at ~1.5-1.6x t_if_covered and falls off a cliff
# past 1.8x as the large items cross the CAP_MULT ceiling; the ranking holds under a
# pessimistic acceptance curve (p0 .35, k 2), so it is not the k<1 rail. Taking the bias
# rather than the fitted peak keeps the value tracking the market as c2f.calibrate refits.
B_QUANTILE = 1.0 / 3.0  # DERIVED, not tuned: accepting costs a, refusing costs 1.5a * P(fair),
# so accept iff P(t >= a) > 2/3, i.e. b = Q(1/3) - PROVIDED the belief's quantiles are honest.
# The tuned values that lived here (0.3333 -> 0.40) were patches for a mis-centred belief;
# the centring now happens in mu (live_shift, fitted by c2f.calibrate on live post-epoch
# labels and monitored there), so the quantile goes back to the number the math gives.
RISK_AVERSION = 0.0  # pure expected value. The mean - lambda*sd shading priced each item as
# if its variance were borne alone, but a round is a portfolio: ~10-20 items x 16 opponents,
# and the round-level sd grows only as sqrt(N) while the shading forfeits mean income
# linearly. Measured over rounds 30-44: shipped charges sat at ~Q(0.27) of the belief and the
# median a/t_lo on proven-fair charges was 0.60-0.79 - a fifth to two fifths of provably
# collectable income forfeited per charge, x16. The knob stays so c2f.autotune/deviation can
# sweep it back up if the market hardens.
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

#: expected payout per opponent when a charge crosses t, as a fraction of the TRUE t.
#: Measured on rounds 38-43 with refusals reconstructed (a recovered as the max payout across
#: reviewers): revenue per opponent is FLAT at 0.48-0.53 * t for every overcharge ratio from
#: 1.0 to 5 - the fitted p0 * r^-k curve (k=0.6 < 1) instead claimed revenue RISES with r,
#: which is an artifact of fitting only the accepted (visible) tail. With a flat step there
#: is no charge-the-moon rail to defend against: R(a) = a * P(t >= a) + 0.5 * E[t | t < a]
#: has an interior maximum by construction.
FRAUD_PAYOUT_FRAC = 0.5
#: never charge above this belief quantile even where the step optimum sits higher: the flat
#: 0.5t plateau is an average over TODAY'S lenient field, and a above it earns its keep only
#: through that plateau holding. Chosen on a TWO-SIDED counterfactual (income from truth
#: brackets, expense against the charges actually billed to us) over games 41/42/45 - the
#: c2f.deviation regret can NOT pick this: it costs under-charging only, so it always asks for
#: more. Shipped boards vs new pricing:
#:     q=0.55: net +49k / -17k / +18k, and the count of our charges proven fraudulent is
#:             UNCHANGED from the shipped board in all three rounds;
#:     q=0.65: +73k / -15k / +9k, but game 45's overcharges doubled (5 -> 10);
#:     q=0.75: bigger still on 41 - entirely the whale's unproven upper side (no t_hi).
A_MAX_Q = 0.55
DISAGREEMENT_RATIO = 1.6  # two passes this far apart on t_mid: use the lower one, don't average
CAP_UNCERTAIN_B_SHRINK = 1.0  # was 0.7: extra caution on b when a referenced policy cap's value is
# unknown. Disabled (2026-08-22): on the games 15-41 window 1.0 beats 0.7 at every other setting
# (+1.3k exp, and it cost 4.4k on game 41's whale alone where the flag fired on the biggest item);
# the flag itself still ships to the log for post-mortems.

class InvalidEstimateError(ValueError):
    """A covered+related item reached pricing with t_low=t_mid=t_high<=0 - the SYSTEM
    contract (c2f.llm.SYSTEM) requires a positive t_mid for such items. This must be
    repaired or replaced upstream (c2f.validate + the targeted repair pass in c2f.run),
    never priced here as if the item were uncovered - see the Game 10 item 3 postmortem."""

    def __init__(self, index: object, message: str):
        self.index = index
        super().__init__(message)


BIAS_RANGE = (0.4, 2.5)  # expanded to allow more aggressive category-specific correction
SIGMA_RANGE = (0.15, 1.0)
BETA_RANGE = (0.35, 1.25)  # slope of log t in log t_mid; 1 = the old pure-multiplier model
#: the ABOVE-pivot slope is capped at 1 AT PRICING TIME (the fit stays unconstrained - see
#: c2f.calibrate): the labels up there are mostly censored floors (t_lo-only), so the MLE
#: keeps asking to inflate large estimates (it fits the 1.25 rail), but the money says no -
#: on the games 12-38 window, beta_high 1.25 buys +7k expected for a -61k worse pessimistic,
#: and game 38 (the newest market state) lost on both a and b to the inflation. Above the
#: pivot the flat bias is the aggressive end of what the evidence supports.
BETA_HIGH_RANGE = (0.35, 1.0)
#: third spline segment for WHALE items (t_mid >= WHALE_T): above this knot the model still
#: runs 30-40% low even on the new prompt - game 41's robbery compensation (t_mid 8,000,
#: t proven >= 11,131, the round worth more than several normal ones), game 40 item 12
#: (t in [2137, 2880)), game 10's watch (4,000 vs >= 7,225). Lifting the whole above-pivot
#: range instead (beta_high > 1) drags the 500-1500 mid-range into fraud (games 37/38 flip
#: negative), so the lift starts only at the knot; continuous there by construction.
#: WHALE_SLOPE swept on the games 15-41 backtest window - see the constant note below.
WHALE_T = 2000.0
#: the whale slope is FITTED by c2f.calibrate (Calibration.beta_whale) with the same spline the
#: pricing uses - fitting and pricing with different models double-counts corrections (the art
#: bucket fitted 0.96 while pricing multiplied the same items by a hard-coded 1.3 lift). At
#: pricing time it is clamped to this range: the floor of 1.0 is the money rail (never shrink a
#: whale below the flat bias - refusing fair charges on the biggest item of a round is the
#: catastrophic side, game 41 = 76k of penalties; over-estimated whales are the art bucket's
#: job); the 1.4 ceiling bounds extrapolation beyond the swept range.
WHALE_RANGE = (1.0, 1.4)
#: t_mid where the affine correction equals `bias` alone: t_hat = bias * mid * (mid/PIVOT)^(beta-1).
#: Chosen near the geometric centre of the labelled t_mids so `bias` keeps its old meaning there.
PIVOT_T = 150.0
LIVE_SHIFT_RANGE = (-0.5, 0.5)  # +-65% multiplicative; a live drift beyond that means refit, not shift
P0_RANGE = (0.02, 0.8)
K_RANGE = (0.0, 6.0)
#: description keyword -> bucket. First match wins, so order matters: the specific
#: trades come before the generic "labour" and "material" catch-alls.
BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    # art/specialist FIRST: these descriptions also contain "assess"/"restor"/"transport"
    # words that other buckets would claim, and their price level is a different market
    # (conservators, art handlers) whose calibration error must not bleed into building
    # repairs - the games 40/42 fine-art overshoot dragged the GLOBAL bias to 0.83 when
    # these items sat in restoration/repair.
    ("art/specialist", ("painting", "artwork", "art restor", "antique", "sculpture",
                        "conservator", "gallery", "museum")),
    ("ancillary/call-out", ("vehicle", "travel", "call-out", "callout", "shipping", "delivery",
                            "disposal", "admin", "postage", "mileage")),
    ("diagnostic/inspect", ("diagnos", "inspect", "assess", "report", "survey", "measurement",
                            "moisture", "leak detection", "detection")),
    ("drying/remediation", ("drying", "dehumid", "dryer", "fan", "vacuum", "extract",
                            "stabilis", "stabiliz")),
    ("electronics", ("tv", "television", "speaker", "laptop", "computer", "monitor", "console",
                     "camera", "phone", "audio", "electronic", "appliance")),
    ("valuables", ("watch", "jewel", "ring", "gold", "bicycle", "bike", "sunglasses", "designer",
                   "compensation", "robbery", "theft", "stolen", "burglar")),
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
    bias: float = 1.0  # true t / model t_mid, median, measured at t_mid = PIVOT_T
    sigma: float = 0.4  # log-sd of true t around the corrected median
    #: slope of log t in log t_mid, BELOW the pivot. The labels show the error is size-shaped,
    #: not a constant multiplier: t_mid < 100 runs ~1.5x low while t_mid >= 400 runs closer to
    #: fair, and a pure bias cannot express that - it is why the fitted sigma railed at the 1.0
    #: clamp. With beta < 1 small estimates are pulled up:
    #:   median t_hat = bias * t_mid * (t_mid / PIVOT_T) ** (beta - 1)   for t_mid < PIVOT_T
    beta: float = 1.0
    #: slope ABOVE the pivot, fitted separately (a linear spline in log space, knot at PIVOT_T,
    #: continuous there by construction). One global slope let the mass of small bracketed items
    #: drag beta to 0.65 and crushed the rare big-ticket items - game 10's watch (t proven
    #: >= 7225, model 4000) priced as if t were ~1800, refusing every fair charge at 1.5x.
    beta_high: float = 1.0
    #: slope of the third spline segment, above WHALE_T - see WHALE_RANGE.
    beta_whale: float = 1.0
    #: additive log-space correction fitted ONLY on live post-epoch rounds (c2f.calibrate).
    #: The main fit runs on current-prompt REPLAYS for breadth, but a replay is not a live
    #: round (chunking, latency, resampling differ) and the live reliability check measured
    #: the replay-fitted belief 22% HIGH on rounds 41-42 while the same belief ran LOW on the
    #: old era. One parameter, interval-censored MLE, everything else held fixed - small
    #: enough to be estimable from few live labels, and it is what keeps B_QUANTILE = 1/3
    #: honest.
    live_shift: float = 0.0
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
        beta = float(d.get("beta", 1.0))  # absent in files written before the affine fit
        beta_high = float(d.get("beta_high", 1.0))
        beta_whale = float(d.get("beta_whale", 1.0))
        live_shift = float(d.get("live_shift", 0.0))
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT_CALIBRATION
    if not all(math.isfinite(v) for v in (*vals.values(), beta, beta_high, beta_whale, live_shift)):
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
        beta=_clamp(beta, BETA_RANGE),
        beta_high=_clamp(beta_high, BETA_HIGH_RANGE),
        beta_whale=_clamp(beta_whale, WHALE_RANGE),
        live_shift=_clamp(live_shift, LIVE_SHIFT_RANGE),
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
        # between-sample disagreement from llm.resample_whales: widens the belief exactly on
        # the (dominant) items where independent samples of the same prompt landed apart
        model_sigma = max(model_sigma, _num(est.get("_sample_sigma")))
        # linear spline in log space, knots at PIVOT_T and WHALE_T: below the pivot beta < 1
        # pulls small t_mids up (the labelled errors are size-shaped - see Calibration.beta);
        # between the knots beta_high applies; above WHALE_T the whale lift comes on top
        # (see WHALE_T). Continuous at both knots. All slopes 1 = the old pure-bias model.
        lm = math.log(mid)
        mu = math.log(mid * cal.bias_for(est.get("_description")))
        if mid < PIVOT_T:
            mu += (cal.beta - 1.0) * (lm - math.log(PIVOT_T))
        else:
            mu += (cal.beta_high - 1.0) * (min(lm, math.log(WHALE_T)) - math.log(PIVOT_T))
            if mid >= WHALE_T:
                mu += (cal.beta_whale - 1.0) * (lm - math.log(WHALE_T))
        # centre on what LIVE rounds realised, not what the replay store fitted (see
        # Calibration.live_shift). Applied last: it is a correction to the whole stack.
        mu += cal.live_shift
        return cls(mu=mu, sigma=_clamp(max(cal.sigma, model_sigma), SIGMA_RANGE))


def accept_limit(belief: Belief) -> float:
    return belief.quantile(B_QUANTILE)


def best_charge(belief: Belief, cal: Calibration, n_grid: int = N_GRID, risk_aversion: float | None = None) -> float:
    """Charge maximising expected per-opponent revenue under the measured payoff step.

    A charge at or under t is paid by EVERY reviewer (accepted pays a, refused-but-fair still
    pays a plus the refuser's own 0.5a penalty), so revenue below t is exactly a. A charge
    over t pays a flat FRAUD_PAYOUT_FRAC * t per opponent on average - measured flat in the
    overcharge ratio, see the constant. So per opponent:

        R(a) = a * P(t >= a) + FRAUD_PAYOUT_FRAC * E[t ; t < a]

    Both terms are closed-form under the lognormal belief; the grid keeps the code shape and
    lets a risk_aversion sweep still bite if c2f.autotune ever turns it back on. The old
    p0 * r^-k acceptance curve is gone from the objective: k < 1 made modelled revenue RISE
    with the overcharge ratio forever (a censoring artifact), so the optimiser only worked
    because a never-above-the-median rail overrode it. The step's optimum is interior; the
    A_MAX_Q cap bounds how hard we lean on the lenient-field plateau."""
    if risk_aversion is None:
        risk_aversion = RISK_AVERSION
    ts = [belief.quantile((j + 0.5) / n_grid) for j in range(n_grid)]
    a_cap = belief.quantile(A_MAX_Q)
    best_a, best_v = 0.0, -math.inf
    for a in ts:
        if a > a_cap:
            break
        pays = [a if a <= t else FRAUD_PAYOUT_FRAC * t for t in ts]
        mean = sum(pays) / n_grid
        v = mean
        if risk_aversion:
            sd = math.sqrt(sum((x - mean) ** 2 for x in pays) / n_grid)
            v -= risk_aversion * sd
        if v > best_v:
            best_a, best_v = a, v
    return best_a


def _has_bundle_keywords(description: str | None) -> bool:
    """Detect likely bundled items with multiple distinct components."""
    if not description:
        return False
    d = (description or "").lower()
    # Common bundling patterns: multiple appliances, multiple plumbing components, etc.
    bundle_indicators = [
        ("boiler", "tank", "flue", "pipework", "heater", "valve"),
        ("repair", "replacement", "adjustment"),
    ]
    # Check if description mentions structural terms suggesting bundled work
    component_keywords = [
        "boiler", "flue", "tank", "pipework", "heater", "valve", "radiator",
        "pipe", "fitting", "gasket", "thermostat", "pump",
        "installation", "removal", "adjustment", "alignment",
    ]
    keyword_count = sum(1 for kw in component_keywords if kw in d)
    return keyword_count >= 2


def _validate_bundle_coverage(est: dict) -> bool:
    """For bundled items marked covered, verify component coverage is explicit.

    If an item description mentions multiple components/operations and is marked
    covered, the reason/clause should explicitly mention that each component is
    covered. If coverage is unclear for any component, return False to mark the
    entire bundle as uncovered (never invent allocations).
    """
    if not est.get("covered"):
        return True  # Not covered, no need to validate bundle

    description = est.get("_description", "") or est.get("description", "")
    if not _has_bundle_keywords(description):
        return True  # Not a bundle, normal coverage applies

    reason = (est.get("reason") or "").lower()
    clause = (est.get("clause") or "").lower()
    evidence_text = (reason + " " + clause).lower()

    # Check for negative indicators: if the reason mentions uncertainty or exclusions
    # for any component, the entire bundle is uncovered
    uncertainty_phrases = [
        "uncertain", "unclear", "not established", "not clear", "potentially",
        "may be excluded", "may not be", "questionable", "disputed",
    ]
    for phrase in uncertainty_phrases:
        if phrase in evidence_text:
            return False

    # For a bundled replacement, all key components must be explicitly mentioned
    # as covered. Extract components from description and verify coverage mention.
    component_keywords = {
        "boiler": ("boiler", "heating", "furnace"),
        "flue": ("flue", "chimney", "vent"),
        "tank": ("tank", "vessel", "cylinder", "storage"),
        "pipework": ("pipe", "pipework", "plumbing", "fitting"),
        "installation": ("install", "fit", "fit", "assemble"),
        "removal": ("remov", "strip", "dismantle"),
        "adjustment": ("adjust", "align", "balance"),
    }

    description_lower = description.lower()
    mentioned_components = []
    for comp_name, keywords in component_keywords.items():
        if any(kw in description_lower for kw in keywords):
            mentioned_components.append(comp_name)

    # If multiple components are mentioned, all must be explicitly covered in evidence
    if len(mentioned_components) >= 2:
        # Check that the reason/clause mentions the key operation being covered
        covered_phrases = [
            "covered", "indemnifi", "reimburse", "claim", "payable", "insured",
        ]
        has_coverage_language = any(phrase in evidence_text for phrase in covered_phrases)
        if not has_coverage_language:
            return False

    return True


def price_item(est: dict, cal: Calibration | None = None, other: dict | None = None) -> tuple[float, float]:
    """Return (a, b) as gross totals, both finite and >= 0.

    `other`, if given, is a second independent estimate for the same item - see module
    docstring for how a coverage split or a wide t_mid gap between the two is handled."""
    cal = cal or calibration()
    covered = bool(est.get("covered", False)) and bool(est.get("related", True))

    # Gate: validate bundle coverage BEFORE any pricing. If bundle coverage is not
    # established (evidence doesn't cover all components), treat as truly uncovered.
    bundle_coverage_valid = True
    if covered:
        bundle_coverage_valid = _validate_bundle_coverage(est)
        if not bundle_coverage_valid:
            covered = False

    mid = sorted(_num(est.get(k)) for k in ("t_low", "t_mid", "t_high"))[1]

    other_covered = other_mid = None
    other_bundle_valid = True
    if other is not None:
        other_covered = bool(other.get("covered", False)) and bool(other.get("related", True))
        if other_covered:
            other_bundle_valid = _validate_bundle_coverage(other)
            if not other_bundle_valid:
                other_covered = False
        other_mid = sorted(_num(other.get(k)) for k in ("t_low", "t_mid", "t_high"))[1]

    if other_covered is not None and other_covered != covered:
        # split coverage call: not confident enough either way to pay out
        guess = _num(est.get("t_if_covered")) or mid or other_mid or _num(other.get("t_if_covered"))
        return round(UNCOVERED_CHARGE * guess, 2), 0.0

    if not covered:
        # For items failing bundle validation, return a=0, b=0 to enforce strict
        # coverage. For normal uncovered items, apply UNCOVERED_CHARGE.
        if bundle_coverage_valid == False and bool(est.get("covered", False)):
            # Model said covered but validation rejected due to bundle components
            return 0.0, 0.0
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
    # WHALE FLOOR on b: for the round's biggest items the quantile rule has the asymmetry
    # backwards. Refusing costs 0.5 * a_j per FAIR charge billed to us - driven by the other
    # 16 teams' charges, heavy-tailed in t and unbounded by our belief - while accepting fraud
    # is capped near CAP_MULT * t; and the field's billed fair:fraud value ratio measured ~10:1
    # over rounds 30-44, so the prior on a big charge being fair is high. A multiplicative b
    # also has an ABSOLUTE error that scales with t (Q(1/3) sits t * (1 - e^{-0.43 sigma})
    # under the median - thousands of euros on a whale, at the same sigma that costs a few
    # euros on a small item). Game 41 item 3 alone: 24,363 in penalties from a whale priced
    # by the plain quantile. On whales, refuse only what even the belief's MEDIAN calls fraud.
    if covered and belief.median >= WHALE_T:
        b = max(b, belief.median)
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
