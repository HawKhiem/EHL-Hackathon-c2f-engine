"""Combine several model answers for the same case into one estimate per item.

covered  : majority vote; a tie counts as NOT covered (wrongly paying fraud costs 1x on
           every opponent, wrongly rejecting costs 0.5x extra - but a tie means we are
           far below the 2/3 confidence the b-rule needs).
t_*      : median over the votes that said covered.
t_if_covered : median over all votes that gave one.
"""

from __future__ import annotations

from statistics import median


def _num(x) -> float:
    try:
        v = float(x)
        return v if v == v and v > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def aggregate(outputs: list[dict]) -> dict:
    """outputs: list of model JSONs ({policy_summary, items:[...]}) -> one JSON of same shape."""
    votes: dict[int, list[dict]] = {}
    for out in outputs:
        for it in out.get("items", []):
            try:
                idx = int(it["index"])
            except (KeyError, TypeError, ValueError):
                continue
            votes.setdefault(idx, []).append(it)
    items = []
    for idx in sorted(votes):
        vs = votes[idx]
        yes = [v for v in vs if bool(v.get("covered")) and bool(v.get("related", True))]
        covered = len(yes) * 2 > len(vs)
        if covered:
            lo = median(_num(v.get("t_low")) for v in yes)
            mid = median(_num(v.get("t_mid")) for v in yes)
            hi = median(_num(v.get("t_high")) for v in yes)
            src = yes[0]
        else:
            lo = mid = hi = 0.0
            src = next((v for v in vs if not bool(v.get("covered"))), vs[0])
        ifc_vals = [_num(v.get("t_if_covered")) for v in vs if _num(v.get("t_if_covered"))]
        ifc_vals += [_num(v.get("t_mid")) for v in yes] if not covered else []
        items.append(
            {
                "index": idx,
                "covered": covered,
                "related": covered or bool(src.get("related", True)),
                "clause": src.get("clause", ""),
                "t_low": lo,
                "t_mid": mid,
                "t_high": hi,
                "t_if_covered": median(ifc_vals) if ifc_vals else 0.0,
                "reason": src.get("reason", ""),
                "votes": f"{len(yes)}/{len(vs)} covered",
            }
        )
    return {"policy_summary": outputs[0].get("policy_summary", "") if outputs else "", "items": items, "n_votes": len(outputs)}
