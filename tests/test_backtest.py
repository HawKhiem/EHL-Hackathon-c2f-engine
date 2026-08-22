from c2f.backtest import old_games, verdict


def row(exp_rank: int, exp_net: float = 100.0) -> dict:
    return {"actual_net": 0.0, "pess_net": 0.0, "exp_net": exp_net, "opt_net": 0.0,
            "pess_rank": 9, "exp_rank": exp_rank, "opt_rank": 1}


def test_majority_is_over_all_old_games_not_just_replayed():
    games = {3: row(1)}  # only one game replayed and it won
    t = verdict(games, old=[1, 2, 3, 4, 5])
    assert t["n_games"] == 5 and t["n_scored"] == 1
    assert t["missing"] == [1, 2, 4, 5]
    assert t["success"] is False


def test_top_three_counts_but_a_loss_does_not():
    old = [1, 2, 3, 4, 5]
    # 3rd place with money in the bank is a good game; 1st place while losing money is not
    assert verdict({g: row(3) for g in old}, old)["success"] is True
    assert verdict({g: row(1, exp_net=-1.0) for g in old}, old)["success"] is False
    assert verdict({g: row(4) for g in old}, old)["success"] is False  # profitable but 4th


def test_a_big_win_does_not_excuse_losing_rounds():
    old = [1, 2, 3, 4, 5]
    games = {1: row(1, exp_net=10_000.0)}
    games.update({g: row(1, exp_net=-100.0) for g in old[1:]})
    t = verdict(games, old)
    assert t["expected"] > 0  # total is healthy...
    assert t["good"] == 1 and t["success"] is False  # ...but only 1 of 5 rounds actually paid


def test_success_needs_strict_majority_and_no_missing():
    old = [1, 2, 3, 4]
    assert verdict({g: row(1 if g <= 2 else 9) for g in old}, old)["success"] is False  # 2/4
    assert verdict({g: row(1 if g <= 3 else 9) for g in old}, old)["success"] is True  # 3/4
    t = verdict({g: row(1) for g in [1, 2, 3]}, old)  # 3 good games but one missing
    assert t["good"] == 3 and t["success"] is False and t["missing"] == [4]


def test_empty_population_is_not_success():
    assert verdict({}, [])["success"] is False


def test_old_games_is_last_window_of_decrypted_completed_games(tmp_path, monkeypatch):
    import c2f.backtest as bt
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    for g in (1, 2, 3, 4, 6, 7, 8):  # 5 completed but not decrypted
        (tmp_path / "cases" / f"case_{g:02d}").mkdir(parents=True)
        (tmp_path / "cases" / f"case_{g:02d}" / "policy.txt").write_text("x")
    assert old_games([1, 2, 3, 4, 5, 6, 7, 8]) == [3, 4, 6, 7, 8]
    assert old_games([1, 2, 3]) == [1, 2, 3]


def test_reprice_uses_stored_estimate_and_current_pricing(tmp_path, monkeypatch):
    import json
    import c2f.backtest as bt
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    case_dir = tmp_path / "cases" / "case_09"
    case_dir.mkdir(parents=True)
    (case_dir / "policy.txt").write_text("p")
    (case_dir / "description.txt").write_text("d")
    monkeypatch.setattr(bt, "load_case", lambda d, g: {"game_id": g, "items": [{"index": 1}, {"index": 2}], "images": []})
    est = {"items": [{"index": 1, "covered": True, "related": True, "t_low": 90, "t_mid": 100, "t_high": 110}]}
    rep = bt.reprice(9, {"rows": [{"index": 1, "charge_price": 999, "acceptance_limit": 999}], "estimate": est})
    rows = {r["index"]: r for r in rep["rows"]}
    assert rep["repriced"] is True
    assert 0 < rows[1]["charge_price"] < 100 and 0 < rows[1]["acceptance_limit"] < 100
    assert rows[2] == {"index": 2, "charge_price": 0.0, "acceptance_limit": 0.0}  # missing from the model -> unknown
    assert bt.reprice(9, {"rows": []}) == {"rows": []}  # nothing stored to re-price
