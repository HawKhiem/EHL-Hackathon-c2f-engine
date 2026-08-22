"""Manual mode (no LLM key): `show` prints the case compactly, `go` prices+submits a spec.

  pixi run python -m c2f.manual show 2            # fetch (polls key) + print compact case
  pixi run python -m c2f.manual go 2 "1:80:150:250 2:150:250:400 3:x:1500 4:x"
     spec per item:  IDX:LOW:MID:HIGH   (covered)   |  IDX:x[:t_if_covered]  (not covered)
"""

from __future__ import annotations

import json
import re
import sys
import time

from c2f.extract import load_case
from c2f.price import price_all
from c2f.submit import ROOT, fetch_case, submit

KEYWORDS = re.compile(
    r"(exclu|not covered|not insured|limit|deductible|excess|ancillar|call-out|travel|vehicle|"
    r"upgrade|betterment|improve|source installation|pipe|leak detection|trace|drying|"
    r"consequential|sum insured|cap|maximum|%|per cent|EUR|€)",
    re.I,
)


def show(game_id: int) -> None:
    t0 = time.time()
    case_dir = ROOT / "cases" / f"case_{game_id:02d}"
    if not (case_dir / "policy.txt").exists():
        case_dir = fetch_case(game_id)
    c = load_case(case_dir, game_id)
    print(f"### key+decrypt {time.time()-t0:.1f}s   images={len(c['images'])}")
    print("### DESCRIPTION\n" + c["description"])
    print("### ITEMS")
    if c["items"]:
        for it in c["items"]:
            print(f"  {it['index']:>2} | {it['description']} | {it['quantity']:g} {it['unit']}")
    else:
        print(c["invoice_text"])
    print("### POLICY (headings + lines with money/limit/exclusion words)")
    for ln in c["policy"].splitlines():
        s = ln.strip()
        if not s or set(s) <= set("-="):
            continue
        if re.match(r"^(PART|\d+(\.\d+)*\s)", s) or KEYWORDS.search(s):
            print("  " + s[:160])
    print(f"### shown at {time.time()-t0:.1f}s")


def go(game_id: int, spec: str) -> None:
    est = []
    for tok in spec.split():
        p = tok.split(":")
        idx = int(p[0])
        if p[1].lower() == "x":
            est.append({"index": idx, "covered": False, "related": False, "t_low": 0, "t_mid": 0, "t_high": 0,
                        "t_if_covered": float(p[2]) if len(p) > 2 else 0})
        else:
            lo, mid, hi = (float(v) for v in p[1:4])
            est.append({"index": idx, "covered": True, "related": True, "t_low": lo, "t_mid": mid, "t_high": hi})
    rows = price_all(est)
    resp = submit(game_id, rows)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"game_{game_id:02d}.json").write_text(
        json.dumps({"game_id": game_id, "manual": True, "est": est, "rows": rows, "response": resp}, indent=1)
    )
    print(f"OK {len(resp)} rows: " + " ".join(f"#{r['index']}={r['charge_price']}/{r['acceptance_limit']}" for r in rows))


if __name__ == "__main__":
    cmd, gid = sys.argv[1], int(sys.argv[2])
    if cmd == "show":
        show(gid)
    elif cmd == "go":
        go(gid, sys.argv[3])
    else:
        sys.exit(__doc__)
