import c2f.policy as policy
from c2f.extract import MAX_CHARS, load_case
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


def test_short_policy_is_not_capped_and_needs_no_digest(tmp_path):
    case = _case(tmp_path, "x" * (MAX_CHARS - 1))
    assert case["truncated"] == []
    assert policy.attach(case, tmp_path) is None
    assert "policy_digest" not in case


def test_long_policy_is_capped_and_digest_is_attempted(tmp_path, monkeypatch):
    case = _case(tmp_path, "x" * (MAX_CHARS + 10))
    assert case["truncated"] == ["policy"]
    assert case["policy"].endswith("[... truncated ...]")

    seen = {}

    def fake_distill(text, *, timeout=0.0, model=None):
        seen["chars"] = len(text)
        return "EXCLUSIONS:\n- flood, cl. 7", {"model": "fake"}

    monkeypatch.setattr(policy, "distill", fake_distill)
    assert policy.attach(case, tmp_path) == {"model": "fake"}
    # the digest reads the WHOLE file, not the capped copy
    assert seen["chars"] == MAX_CHARS + 10
    assert "flood, cl. 7" in build_user_message(case)
    assert "<policy_digest" in build_user_message(case)


def test_digest_failure_leaves_the_case_usable(tmp_path, monkeypatch):
    case = _case(tmp_path, "x" * (MAX_CHARS + 10))

    def boom(text, *, timeout=0.0, model=None):
        raise RuntimeError("timeout")

    monkeypatch.setattr(policy, "distill", boom)
    assert policy.attach(case, tmp_path) == {"error": "timeout"}
    assert "policy_digest" not in case
    assert "<policy_digest" not in build_user_message(case)
