"""Structured validation of model item output against the SYSTEM contract (c2f.llm.SYSTEM).

A covered AND related item must carry a positive, finite t_mid (and a sane t_low/t_high
around it); the zero convention (t_low=t_mid=t_high=0) is reserved for not-covered /
not-related items. An all-zero covered+related item is a broken answer, not a cheap
estimate - it must never reach c2f.price as if it were an ordinary $0 item (see the
Game 10 item 3 postmortem: this exact shape caused a wrongful-rejection penalty on a
watch proven worth >= EUR 7,225).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

REQUIRED_FIELDS = ("index", "covered", "related", "t_low", "t_mid", "t_high")


@dataclass(frozen=True)
class ValidationError:
    index: int | None
    field: str
    message: str

    def __str__(self) -> str:
        return f"item {self.index}: {self.message}" if self.index is not None else self.message


def _finite(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def validate_items(out: dict, expected_indices: set[int] | None = None) -> list[ValidationError]:
    """Every violation of the SYSTEM contract in out['items']. Empty list = fully valid."""
    items = out.get("items") if isinstance(out, dict) else None
    if not isinstance(items, list):
        return [ValidationError(None, "items", "no items list in model output")]

    errors: list[ValidationError] = []
    seen: set[int] = set()
    for raw in items:
        if not isinstance(raw, dict):
            errors.append(ValidationError(None, "item", "item is not an object"))
            continue

        try:
            idx = int(raw.get("index"))
        except (TypeError, ValueError):
            errors.append(ValidationError(None, "index", f"invalid index {raw.get('index')!r}"))
            continue
        if idx in seen:
            errors.append(ValidationError(idx, "index", "duplicate index"))
        seen.add(idx)

        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        for f in missing:
            errors.append(ValidationError(idx, f, f"missing required field {f!r}"))
        if missing:
            continue  # nothing left to numerically validate

        covered = bool(raw.get("covered")) and bool(raw.get("related", True))
        if not covered:
            continue  # zero valuation is legitimate for uncovered/unrelated items

        t_low, t_mid, t_high = (_finite(raw.get(k)) for k in ("t_low", "t_mid", "t_high"))
        if t_mid is None or t_mid <= 0:
            errors.append(
                ValidationError(idx, "t_mid", f"covered+related item must have finite t_mid > 0, got {raw.get('t_mid')!r}")
            )
            continue
        if t_high is None or t_high < t_mid:
            errors.append(
                ValidationError(idx, "t_high", f"t_high must be finite and >= t_mid, got {raw.get('t_high')!r}")
            )
        if t_low is None or not (0 <= t_low <= t_mid):
            errors.append(
                ValidationError(idx, "t_low", f"t_low must be finite and in [0, t_mid], got {raw.get('t_low')!r}")
            )

    if expected_indices is not None:
        for idx in sorted(expected_indices - seen):
            errors.append(ValidationError(idx, "index", "expected index missing from output"))

    return errors


def invalid_indices(errors: list[ValidationError]) -> set[int]:
    return {e.index for e in errors if e.index is not None}
