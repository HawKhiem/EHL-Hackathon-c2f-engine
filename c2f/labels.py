"""One place to read what a finished game left behind.

Every analysis tool (c2f.accuracy, c2f.calibrate, c2f.deviation, c2f.history, c2f.postmortem)
joins the same three things: the model's per-item estimate, the (a, b) board that actually went
out, and the t brackets c2f.truth recovered from the leaderboard. They used to each carry their
own copy of that loading - four `estimates()` functions with two different key precedences
between them, which is a disagreement waiting to matter the day a log carries both keys.
"""

from __future__ import annotations

import json

from c2f.price import bucket_of
from c2f.submit import ROOT

RUNS = ROOT / "runs"

#: run-log keys holding the model's per-item output, newest shape first. `estimate` is what
#: c2f.run writes today; `ensemble` is from when several votes were aggregated; the
#: `model_full*` outputs are the raw pass, used when a run died before it merged one.
ESTIMATE_KEYS = ("estimate", "ensemble", "model_full", "model_full0")


def log(game_id: int) -> dict:
    """The run log for a game (the real one, else a dry run). {} if there is none."""
    for name in (f"game_{game_id:02d}.json", f"dry_game_{game_id:02d}.json"):
        p = RUNS / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def estimates_from(rec: dict) -> dict[int, dict]:
    """Per-item model estimate out of an already-loaded run log, indexed by POS.

    The invoice wording rides along as `_description`: the model's own output has no
    description field, and the category is what c2f.price's per-bucket bias keys on.
    """
    src: dict = {}
    for key in ESTIMATE_KEYS:
        v = rec.get(key) or {}
        v = (v.get("output") or {}) if key.startswith("model_") else v
        if v.get("items"):
            src = v
            break
    items = {int(i["index"]): i for i in src.get("items", []) if "index" in i}
    descs = {int(i["index"]): i.get("description", "") for i in rec.get("case", {}).get("items", [])}
    for i, it in items.items():
        it.setdefault("_description", descs.get(i, ""))
    return items


def estimates(game_id: int) -> dict[int, dict]:
    return estimates_from(log(game_id))


def board(game_id: int) -> dict[int, dict]:
    """The (a, b) per item as the server saw them at close: submissions applied in order,
    last write wins - the same rule the scoring uses."""
    rows: dict[int, dict] = {}
    for sub in log(game_id).get("submissions") or []:
        for r in sub.get("rows") or []:
            try:
                rows[int(r["index"])] = r
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def truths(games: list[int] | None = None) -> dict[int, dict]:
    """game id -> {item index: {t_lo, t_hi, ...}} for every game c2f.truth has labelled."""
    out: dict[int, dict] = {}
    for p in sorted(RUNS.glob("truth_game_*.json")):
        g = int(p.stem.split("_")[-1])
        if games is not None and g not in games:
            continue
        out[g] = {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    return out


def rows(games: list[int] | None = None) -> list[dict]:
    """One row per labelled item: what we estimated against what the market proved.

    `t_hi` is None when the bracket is open above. Callers filter: c2f.calibrate drops rows
    that prove nothing (t_lo <= 0 and no t_hi), c2f.accuracy keeps them to count abstentions.
    """
    out: list[dict] = []
    for g, truth in truths(games).items():
        est = estimates(g)
        for i, tv in truth.items():
            it = est.get(i)
            if not it:
                continue
            try:
                mid = float(it.get("t_mid") or 0)
            except (TypeError, ValueError):
                mid = 0.0
            t_hi = tv.get("t_hi")
            out.append({
                "game": g,
                "item": i,
                "bucket": bucket_of(it.get("_description", "")),
                "description": it.get("_description", ""),
                "covered": bool(it.get("covered")) and bool(it.get("related", True)),
                "t_mid": mid,
                "t_lo": float(tv.get("t_lo") or 0.0),
                "t_hi": float(t_hi) if t_hi is not None else None,
            })
    return out
