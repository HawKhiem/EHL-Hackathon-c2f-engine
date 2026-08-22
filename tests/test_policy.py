import c2f.policy as policy
from c2f.extract import load_case
from c2f.llm import build_user_message


def test_render_skips_empty_sections():
    out = policy.render({"insured_event": "fire", "limits": ["EUR 5000 cap, cl. 4"], "exclusions": []})
    assert "INSURED EVENT: fire" in out
    assert "- EUR 5000 cap, cl. 4" in out
    assert "EXCLUSIONS" not in out


def test_render_empty_digest_is_empty():
    assert policy.render({"insured_event": "", "limits": [], "exclusions": None}) == ""


def _case(dirpath, policy_text):
    (dirpath / "policy.txt").write_text(policy_text)
    (dirpath / "description.txt").write_text("a pipe burst")
    return load_case(dirpath, 1)


def test_a_long_policy_reaches_the_model_whole(tmp_path):
    body = "clause 1\n" + "x" * 200_000 + "\n17. EXCLUSIONS: flood"
    case = _case(tmp_path, body)
    assert case["policy"] == body
    assert "truncated" not in case
    assert "[... truncated ...]" not in build_user_message(case)
    assert "17. EXCLUSIONS: flood" in build_user_message(case)


def test_build_reads_the_whole_file_and_attach_assigns(tmp_path, monkeypatch):
    case = _case(tmp_path, "x" * 90_000)
    seen = {}

    def fake_distill(text, *, timeout=0.0, model=None):
        seen["chars"] = len(text)
        return "EXCLUSIONS:\n- flood, cl. 7", {"model": "fake"}

    monkeypatch.setattr(policy, "distill", fake_distill)
    assert policy.attach(case, tmp_path) == {"model": "fake"}
    assert seen["chars"] == 90_000
    msg = build_user_message(case)
    assert "<policy_digest" in msg and "flood, cl. 7" in msg


def test_digest_failure_leaves_the_case_usable(tmp_path, monkeypatch):
    case = _case(tmp_path, "policy body")

    def boom(text, *, timeout=0.0, model=None):
        raise RuntimeError("timeout")

    monkeypatch.setattr(policy, "distill", boom)
    text, meta = policy.build(tmp_path)
    assert (text, meta) == (None, {"error": "timeout"})
    assert policy.attach(case, tmp_path) == {"error": "timeout"}
    assert "policy_digest" not in case
    assert "<policy_digest" not in build_user_message(case)
    assert "policy body" in build_user_message(case)


def test_missing_policy_file_is_not_an_error(tmp_path):
    text, meta = policy.build(tmp_path)
    assert text is None and "skipped" in meta
