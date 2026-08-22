"""First real a/b strategy - hardcoded constants, meant to be tested and iterated on, not a
final model.

a = t_estimate * 0.6                     - charge below the fair value so opposing insurers
                                            have no reason to reject it as fraud.
b = t_estimate * m(t_estimate)           - accept generously on cheap items (nobody profitably
                                            filters moderate fraud there anyway) and tightly on
                                            expensive ones (where the payment cap c >= 4t makes
                                            an accepted overcharge expensive). m decays from 4x
                                            on cheap items toward a 1.1x floor on expensive ones.

t_estimate is the arithmetic mean of (t_low, t_high), falling back to t_low when t_high is null.
"""
from __future__ import annotations

import math

A_MULTIPLIER = 0.6
B_DECAY_SCALE = 50.0
B_MULTIPLIER_FLOOR = 1.1
B_MULTIPLIER_CEILING = 4.0


def t_estimate(item: dict) -> float:
    t_low = float(item["t_low"])
    t_high = item["t_high"]
    if t_high is None:
        return t_low
    return (t_low + float(t_high)) / 2


def _b_multiplier(t: float) -> float:
    return max(B_MULTIPLIER_FLOOR, B_MULTIPLIER_CEILING / (1 + math.log10(1 + t / B_DECAY_SCALE)))


def price_item(item: dict) -> tuple[float, float]:
    if not item["covered"]:
        return 0.0, 0.0
    t = t_estimate(item)
    a = t * A_MULTIPLIER
    b = t * _b_multiplier(t)
    return a, b
