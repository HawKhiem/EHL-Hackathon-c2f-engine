"""Core value types for the Claim to Fame pipeline.

These cross every layer boundary, so they hold data only — no LLM calls, no I/O,
no arithmetic beyond trivial accessors. See `docs/DESIGN.md`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

#: The quantile ladder we ask the pricing model for, in order.
QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)

#: The insurer's indifference point, straight off the payoff table:
#: accepting costs `a`, wrongly rejecting costs `1.5a`, so accept iff
#: `a <= 1.5 * a * P(T >= a)`  <=>  `P(T >= a) >= 2/3`.
ACCEPT_THRESHOLD: float = 2.0 / 3.0

#: `c >= 4t`, so charging above 4x our median belief is pure waste even if the
#: item turns out to be worth more than we thought.
MAX_CHARGE_MULTIPLE: float = 4.0


@dataclass(frozen=True, slots=True)
class LineItem:
    """One row of `invoices.pdf`, as parsed. Prices are not given to us."""

    item_id: str
    description: str
    quantity: float = 1.0
    unit: str | None = None
    raw_text: str | None = None


@dataclass(frozen=True, slots=True)
class PriceQuantiles:
    """A price belief as five quantiles.

    Always a **unit** price — the gross total is `quantity * unit`, and that
    multiplication happens in code, never in a prompt.
    """

    q10: float
    q25: float
    q50: float
    q75: float
    q90: float

    @property
    def values(self) -> tuple[float, ...]:
        return (self.q10, self.q25, self.q50, self.q75, self.q90)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> PriceQuantiles:
        if len(values) != len(QUANTILE_LEVELS):
            raise ValueError(f"expected {len(QUANTILE_LEVELS)} quantiles, got {len(values)}")
        return cls(*(float(v) for v in values))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> PriceQuantiles:
        """Tolerant constructor for model output: accepts `q10` or `0.1` keys."""
        out: list[float] = []
        for level in QUANTILE_LEVELS:
            key = f"q{int(round(level * 100))}"
            value = raw.get(key, raw.get(str(level), raw.get(level)))  # type: ignore[arg-type]
            if value is None:
                raise ValueError(f"missing quantile {key} in {sorted(raw)}")
            out.append(float(value))  # type: ignore[arg-type]
        return cls.from_values(out)


@dataclass(frozen=True, slots=True)
class ItemInference:
    """What the semantic layer believes about one line item.

    `p_valid` is asked for directly as P(covered AND related). It is deliberately
    not `p_covered * p_related` — those events are strongly dependent, and the
    product understates validity. The two marginals are kept for display only.
    """

    item_id: str
    p_valid: float
    unit_quantiles: PriceQuantiles
    p_covered: float | None = None
    p_related: float | None = None
    #: Skeptic's shrink on the whole ladder. Clamped to [0.5, 1.0] downstream:
    #: a shrink degrades gracefully where a veto would be brittle.
    skeptic_multiplier: float = 1.0
    evidence: str = ""
    confidence: str = ""
    #: True when this item fell back to a heuristic because inference failed.
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class Calibration:
    """Everything the system learns across rounds. Four numbers, on purpose.

    `mu_shift` / `sigma_scale` correct the price belief; `lambda_a` / `lambda_b`
    correct the decision. An 8-feature classifier cannot be fit from 20 rounds
    of data; these can.
    """

    #: Multiplicative bias on our median. >1 means true thresholds run high.
    mu_shift: float = 1.0
    #: Spread correction in log space. >1 means we are overconfident.
    sigma_scale: float = 1.0
    #: Charge aggressiveness knob.
    lambda_a: float = 1.0
    #: Acceptance generosity knob.
    lambda_b: float = 1.0
    n_obs: int = 0


@dataclass(frozen=True, slots=True)
class ItemDecision:
    """The submitted pair, plus everything needed to reconstruct it.

    `b < a` is legitimate and expected: the optimal charge tracks the spread of
    our belief while the acceptance limit is pinned at the 2/3 confidence point.
    Do not "fix" it.
    """

    item_id: str
    a: float
    b: float
    s_at_a: float
    s_at_b: float
    sigma_log: float
    q50_gross: float
    p_valid: float
    quantity: float = 1.0
    reason: str = ""
    notes: list[str] = field(default_factory=list)
