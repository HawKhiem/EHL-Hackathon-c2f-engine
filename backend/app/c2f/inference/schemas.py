"""Getting structured data out of model output without trusting it.

Two jobs: find the JSON in a reply that may be wrapped in prose or fences, and
coerce whatever came back into `ItemInference` without ever raising. A malformed
reply must degrade one item, never lose the round.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from app.c2f.models import ItemInference, LineItem, PriceQuantiles
from app.c2f.probability.survival import quantiles_from_median

#: Used when the validity call fails outright. Deliberately just above the 2/3
#: bar: a blanket `b = 0` would pay the 1.5x lawyer penalty to every opponent on
#: every genuinely covered item, which is the most expensive way to be wrong.
DEFAULT_P_VALID: float = 0.75
#: Used when the pricing call fails. Wide on purpose - honest uncertainty pushes
#: the charge up and the acceptance limit down, which is the safe direction on
#: both sides.
FALLBACK_MEDIAN: float = 200.0
FALLBACK_SIGMA_LOG: float = 1.2


def _scan_balanced(text: str, start: int, opener: str, closer: str) -> Any | None:
    """Parse the balanced `opener`..`closer` span beginning at `start`.

    String-aware, so a brace inside an evidence string cannot throw off the
    depth count.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_json(text: str) -> Any | None:
    """First complete JSON value in `text`, or None.

    Handles ```json fences, leading commentary, and trailing prose.

    Scans for whichever of `[` or `{` appears **first**, which is the whole
    point: our prompts ask for an array of per-item objects, and an earlier
    version tried `{` before `[`. On `[{...}, {...}]` that returned only the
    first object, so every line item but the first silently fell back to the
    heuristic - on a real 18-item case, 17 wrong prices and no error anywhere.
    """
    if not text:
        return None

    cursor = 0
    while cursor < len(text):
        candidates = [(text.find(c, cursor), c) for c in "[{"]
        candidates = [(pos, c) for pos, c in candidates if pos != -1]
        if not candidates:
            return None
        start, opener = min(candidates)
        parsed = _scan_balanced(text, start, opener, "]" if opener == "[" else "}")
        if parsed is not None:
            return parsed
        cursor = start + 1
    return None


def _as_probability(value: Any, default: float) -> float:
    """Coerce to [0, 1]. Accepts percentages, which models emit unprompted."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number > 1.0:
        number = number / 100.0 if number <= 100.0 else 1.0
    return min(max(number, 0.0), 1.0)


def _as_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def index_by_item(payload: Any) -> dict[str, Mapping[str, Any]]:
    """Normalise a reply into `{item_id: fields}`.

    Accepts the three shapes models actually return: a top-level list, a dict
    with an `items` list, or a dict keyed directly by item id.
    """
    if isinstance(payload, Mapping):
        for key in ("items", "line_items", "results"):
            inner = payload.get(key)
            if isinstance(inner, Sequence) and not isinstance(inner, str | bytes):
                payload = inner
                break
        else:
            if all(isinstance(v, Mapping) for v in payload.values()) and payload:
                return {str(k): v for k, v in payload.items()}  # type: ignore[misc]
            payload = [payload]

    if not isinstance(payload, Sequence) or isinstance(payload, str | bytes):
        return {}

    out: dict[str, Mapping[str, Any]] = {}
    for position, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            continue
        raw_id = entry.get("item_id", entry.get("id", entry.get("line", position + 1)))
        out[str(raw_id).strip()] = entry
    return out


def fallback_inference(item: LineItem, *, reason: str = "inference unavailable") -> ItemInference:
    """A wide, mildly-confident belief. Never blocks a submission."""
    return ItemInference(
        item_id=item.item_id,
        p_valid=DEFAULT_P_VALID,
        unit_quantiles=quantiles_from_median(FALLBACK_MEDIAN, FALLBACK_SIGMA_LOG),
        evidence=reason,
        degraded=True,
    )


def merge_inferences(
    items: Sequence[LineItem],
    validity: Mapping[str, Mapping[str, Any]],
    pricing: Mapping[str, Mapping[str, Any]],
    skeptic: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ItemInference]:
    """One `ItemInference` per parsed line item, in invoice order.

    Every item gets an entry even when a call dropped it, because the submission
    is rejected outright if an item is missing.
    """
    skeptic = skeptic or {}
    out: list[ItemInference] = []

    for item in items:
        item_id = item.item_id
        validity_row = validity.get(item_id)
        pricing_row = pricing.get(item_id)

        if validity_row is None and pricing_row is None:
            out.append(fallback_inference(item, reason="no model output for this item"))
            continue

        p_valid = _as_probability(
            (validity_row or {}).get("p_valid"),
            DEFAULT_P_VALID,
        )

        degraded = False
        if pricing_row is None:
            quantiles = quantiles_from_median(FALLBACK_MEDIAN, FALLBACK_SIGMA_LOG)
            degraded = True
        else:
            try:
                quantiles = PriceQuantiles.from_mapping(pricing_row)
            except (ValueError, TypeError):
                median = _as_float(pricing_row.get("q50", pricing_row.get("median")), 0.0)
                quantiles = quantiles_from_median(median or FALLBACK_MEDIAN, FALLBACK_SIGMA_LOG)
                degraded = True

        skeptic_row = skeptic.get(item_id) or {}
        multiplier = _as_float(
            skeptic_row.get("suggested_multiplier", skeptic_row.get("multiplier")), 1.0
        )

        out.append(
            ItemInference(
                item_id=item_id,
                p_valid=p_valid,
                unit_quantiles=quantiles,
                p_covered=_as_probability((validity_row or {}).get("p_covered"), float("nan"))
                if validity_row and "p_covered" in validity_row
                else None,
                p_related=_as_probability((validity_row or {}).get("p_related"), float("nan"))
                if validity_row and "p_related" in validity_row
                else None,
                skeptic_multiplier=multiplier,
                evidence=str((validity_row or {}).get("evidence", ""))[:600],
                confidence=str((pricing_row or {}).get("confidence", "")),
                degraded=degraded,
            )
        )
    return out
