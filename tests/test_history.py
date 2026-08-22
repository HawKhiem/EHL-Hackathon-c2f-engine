"""The market-history block is a prior we hand the model. It must not contradict itself.

Before this was fixed the block quoted a median floor and a median ceiling taken over
disjoint subsets, so it read "paid up to ~65; refused above ~44" in every bucket, and it
pooled coverage refusals (a t_hi on an item never proven to be worth anything) in with real
price ceilings. Both errors pushed the model's t_mid down, and UNDER_ESTIMATE is our
largest loss cause.
"""

from __future__ import annotations

import c2f.history as H


def obs(t_lo, t_hi, t_mid=100.0, game=1, desc="x", bucket="labour"):
    return {"game": game, "item": 1, "bucket": bucket, "description": desc,
            "covered": True, "t_mid": t_mid, "t_lo": t_lo, "t_hi": t_hi}


class TestBandEndsComeFromTheSameItems:
    def test_two_sided_band_never_inverts(self):
        # 3 items priced high with no ceiling, 3 cheap ones carrying both bounds.
        # The old code took median(t_lo) over the first group and median(t_hi) over the
        # second, printing a ceiling under the floor.
        o = [obs(800, None), obs(900, None), obs(1000, None),
             obs(50, 60), obs(55, 65), obs(60, 70)]
        lo, hi, n = H._band(o)
        assert hi is not None
        assert lo <= hi, f"band inverted: {lo} .. {hi}"
        assert n == 3, "band must be backed only by the items carrying both bounds"
        assert (lo, hi) == (55, 65)

    def test_falls_back_to_one_sided_when_too_few_paired(self):
        o = [obs(800, None), obs(900, None), obs(1000, None), obs(50, 60)]
        lo, hi, n = H._band(o)
        assert hi is None, "one paired item is not enough to quote a range"
        assert (lo, n) == (850, 4)  # median of all four proven floors


class TestCoverageRefusalsAreNotPriceCeilings:
    def test_ceiling_ignores_items_never_proven_worth_anything(self):
        # t_lo == 0 means the market refused it at ANY price - that is a coverage call,
        # not evidence about what the work costs.
        priced = [obs(500, 600), obs(520, 620), obs(540, 640)]
        refused = [obs(0, 30), obs(0, 40), obs(0, 51)]
        lo, hi, n = H._band(priced + refused)
        assert (lo, hi, n) == (520, 620, 3)

    def test_all_refused_reports_no_price_signal(self):
        lo, hi, n = H._band([obs(0, 30), obs(0, 40)])
        assert (lo, hi, n) == (0.0, None, 0)
        assert "no price signal" in H._fmt(lo, hi, n)


class TestRenderedBlock:
    def test_never_prints_a_refused_below_paid_contradiction(self):
        block = H.build()
        if not block:
            return  # no truth files in this checkout
        assert "refused above" not in block
        for line in block.splitlines():
            if "fair value sat between" in line:
                a, b = line.split("~EUR ")[1:3]
                lo = float(a.split()[0])
                hi = float(b.split()[0])
                assert lo <= hi, line

    def test_real_data_bands_are_coherent(self):
        from c2f.accuracy import rows
        import collections
        by = collections.defaultdict(list)
        for r in rows():
            if r["description"].strip():
                by[r["bucket"]].append(r)
        for b, o in by.items():
            lo, hi, n = H._band(o)
            if hi is not None:
                assert lo <= hi, f"{b}: {lo} .. {hi}"


class TestEveryFinishedRoundReachesTheHistory:
    """The failure this guards against was silent: nothing errored, the block just stopped moving.

    load_case stopped parsing the invoice, so every item's _description came back "", every row
    bucketed as "other", and build() - which requires a non-blank description - dropped games 27
    and 28 entirely. The same blank also sent c2f.price to the flat global bias instead of the
    item's category. A game that contributes no label at all is the signature; assert on it.
    """

    def test_no_game_contributes_only_blank_labels(self):
        from c2f.accuracy import rows

        R = rows()
        if not R:
            return  # no truth files in this checkout
        labelled = {r["game"] for r in R if r["description"].strip()}
        blank = sorted({r["game"] for r in R} - labelled)
        assert not blank, f"games whose items carry no line label: {blank}"
