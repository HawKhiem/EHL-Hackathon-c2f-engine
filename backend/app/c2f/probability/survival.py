"""The probability layer: turn a quantile ladder into a survival function.

The one object the decision layer consumes is `FairValueDistribution`, whose
only real job is

    S(a) = P(T >= a)  =  p_valid * P(T_+ >= a)

Everything is interpolated in `log(price)` — repair prices are positive and
right-skewed, so linear interpolation in euros puts mass in the wrong place.

Pure stdlib, no numpy: a survival evaluation is a couple of comparisons and one
multiply, so a 2000-point grid over 20 line items costs single-digit
milliseconds. Monte Carlo (see `decision/`) is the only thing that will need
vectorising, and it is not on the critical path.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from app.c2f.models import QUANTILE_LEVELS, Calibration, ItemInference, PriceQuantiles

_NORMAL = NormalDist()

#: Below this, a "price" is noise. Also the floor that keeps log() defined.
MIN_PRICE: float = 0.01
#: Enforced relative gap between consecutive quantiles. Models return ties and
#: occasionally inversions; a flat segment would make the CDF non-invertible.
_MIN_GAP: float = 1e-3
_MIN_SIGMA: float = 1e-3
#: q90/q10 for a lognormal spans this many sigmas.
_Q10_Q90_SIGMAS: float = 2.0 * 1.2815515655446004


def _sanitise(values: tuple[float, ...]) -> list[float]:
    """Force a positive, strictly increasing ladder.

    Guards the three things model output actually does: zeros, ties, and a
    q75 below the q50.
    """
    out: list[float] = []
    previous = 0.0
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = previous
        if not math.isfinite(value):
            value = previous
        value = max(value, MIN_PRICE, previous * (1.0 + _MIN_GAP))
        out.append(value)
        previous = value
    return out


def _fit_tail(
    low_price: float, low_level: float, high_price: float, high_level: float
) -> tuple[float, float]:
    """Lognormal (mu, sigma) through two quantiles.

    Used only outside the knots. Fitting each tail to its own adjacent pair
    (q10/q25 below, q75/q90 above) makes the tail continuous with the
    interpolated body by construction — F(q10) comes out at exactly 0.10.
    """
    z_low = _NORMAL.inv_cdf(low_level)
    z_high = _NORMAL.inv_cdf(high_level)
    sigma = (math.log(high_price) - math.log(low_price)) / (z_high - z_low)
    sigma = max(sigma, _MIN_SIGMA)
    mu = math.log(low_price) - sigma * z_low
    return mu, sigma


class FairValueDistribution:
    """Our belief about one line item's secret threshold `t`.

    A two-part (hurdle) model: `t = 0` with probability `1 - p_valid`, otherwise
    drawn from the positive part described by the quantile ladder. All prices
    here are **gross totals** — quantity has already been applied.
    """

    __slots__ = ("p_valid", "_log_knots", "_levels", "_lower", "_upper", "median", "sigma_log")

    def __init__(self, p_valid: float, gross_quantiles: tuple[float, ...]) -> None:
        self.p_valid = min(max(float(p_valid), 0.0), 1.0)
        ladder = _sanitise(gross_quantiles)

        self._levels = QUANTILE_LEVELS
        self._log_knots = tuple(math.log(v) for v in ladder)
        self._lower = _fit_tail(ladder[0], QUANTILE_LEVELS[0], ladder[1], QUANTILE_LEVELS[1])
        self._upper = _fit_tail(ladder[-2], QUANTILE_LEVELS[-2], ladder[-1], QUANTILE_LEVELS[-1])

        self.median = ladder[2]
        self.sigma_log = (self._log_knots[-1] - self._log_knots[0]) / _Q10_Q90_SIGMAS

    # ---------- construction ----------

    @classmethod
    def from_inference(
        cls,
        inference: ItemInference,
        *,
        quantity: float = 1.0,
        calibration: Calibration | None = None,
    ) -> FairValueDistribution:
        """Build the gross-total distribution for one item.

        Order matters: unit ladder -> quantity -> skeptic shrink -> calibration.
        The quantity multiply happens here so no prompt is ever asked to do
        arithmetic.
        """
        calibration = calibration or Calibration()
        shrink = min(max(inference.skeptic_multiplier, 0.5), 1.0)
        quantity = quantity if quantity and math.isfinite(quantity) and quantity > 0 else 1.0

        scaled = tuple(v * quantity * shrink for v in inference.unit_quantiles.values)
        return cls(inference.p_valid, cls._calibrate(scaled, calibration))

    @staticmethod
    def _calibrate(gross: tuple[float, ...], calibration: Calibration) -> tuple[float, ...]:
        """Shift the median and stretch the spread around it, in log space."""
        ladder = _sanitise(gross)
        log_median = math.log(ladder[2])
        log_shift = math.log(max(calibration.mu_shift, 1e-6))
        scale = max(calibration.sigma_scale, 1e-3)
        return tuple(
            math.exp(log_shift + log_median + scale * (math.log(v) - log_median)) for v in ladder
        )

    # ---------- the positive part ----------

    def cdf_positive(self, price: float) -> float:
        """`P(T_+ <= price)` — conditional on the item being valid at all."""
        if price <= MIN_PRICE:
            return 0.0
        log_price = math.log(price)

        if log_price <= self._log_knots[0]:
            mu, sigma = self._lower
            return _NORMAL.cdf((log_price - mu) / sigma)
        if log_price >= self._log_knots[-1]:
            mu, sigma = self._upper
            return _NORMAL.cdf((log_price - mu) / sigma)

        for index in range(len(self._log_knots) - 1):
            low, high = self._log_knots[index], self._log_knots[index + 1]
            if log_price <= high:
                weight = (log_price - low) / (high - low)
                p_low, p_high = self._levels[index], self._levels[index + 1]
                return p_low + weight * (p_high - p_low)
        return self._levels[-1]

    def quantile_positive(self, level: float) -> float:
        """Inverse of `cdf_positive`. `level` outside (0, 1) clamps."""
        if level <= 0.0:
            return 0.0
        if level >= 1.0:
            return math.exp(self._upper[0] + self._upper[1] * _NORMAL.inv_cdf(1.0 - 1e-9))

        if level <= self._levels[0]:
            mu, sigma = self._lower
            return math.exp(mu + sigma * _NORMAL.inv_cdf(level))
        if level >= self._levels[-1]:
            mu, sigma = self._upper
            return math.exp(mu + sigma * _NORMAL.inv_cdf(level))

        for index in range(len(self._levels) - 1):
            p_low, p_high = self._levels[index], self._levels[index + 1]
            if level <= p_high:
                weight = (level - p_low) / (p_high - p_low)
                low, high = self._log_knots[index], self._log_knots[index + 1]
                return math.exp(low + weight * (high - low))
        return math.exp(self._log_knots[-1])

    # ---------- what the decision layer uses ----------

    def survival(self, price: float) -> float:
        """`S(a) = P(T >= a)`.

        `S(0) = 1` because `t >= 0` always holds, even for an uncovered item.
        Above zero the hurdle bites: `p_valid` scales the whole curve, which is
        exactly why it cancels out of the optimal charge and yet decides the
        acceptance limit.
        """
        if price <= 0.0:
            return 1.0
        return self.p_valid * (1.0 - self.cdf_positive(price))

    def sample_threshold(self, uniform_valid: float, uniform_level: float) -> float:
        """Draw a `t` from two uniforms. Explicit inputs keep replay reproducible."""
        if uniform_valid >= self.p_valid:
            return 0.0
        return self.quantile_positive(uniform_level)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FairValueDistribution(p_valid={self.p_valid:.3f}, "
            f"median={self.median:.2f}, sigma_log={self.sigma_log:.3f})"
        )


def flat_distribution(
    p_valid: float, median: float, sigma_log: float = 0.6
) -> FairValueDistribution:
    """A lognormal fallback for when pricing inference fails or times out.

    Used by the heuristic submission at T+8 and by any item whose inference call
    does not come back in time.
    """
    median = max(float(median), MIN_PRICE)
    sigma_log = max(float(sigma_log), 0.05)
    ladder = tuple(
        median * math.exp(sigma_log * _NORMAL.inv_cdf(level)) for level in QUANTILE_LEVELS
    )
    return FairValueDistribution(p_valid, ladder)


def quantiles_from_median(median: float, sigma_log: float = 0.6) -> PriceQuantiles:
    """The same lognormal ladder, as a `PriceQuantiles` for logging."""
    median = max(float(median), MIN_PRICE)
    sigma_log = max(float(sigma_log), 0.05)
    return PriceQuantiles.from_values(
        [median * math.exp(sigma_log * _NORMAL.inv_cdf(level)) for level in QUANTILE_LEVELS]
    )
