"""Structured pre-extraction of the clauses in a policy that bind the payout decision.

Nothing is truncated any more - the full policy goes to the model verbatim. This digest is
not a substitute for that text, it is a reading aid: policies run 40-65k chars and bury the
caps, deductibles and exclusions at the end, so we ask the FASTEST model to pull them out
and put them in FRONT of the verbatim text. The expensive pass then starts from the clauses
that decide `covered` instead of having to find them.

Cheap enough to run every game (~11-15 s), but the game has a 60 s clock, so it is
best-effort by design: any failure or timeout means the run proceeds on the verbatim policy
alone, which is still complete. run.py starts it in parallel and bounds the wait.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from c2f.llm import _parse_json, provider

# Fastest OpenAI model. Override with C2F_DIGEST_MODEL.
FAST = {"openai": "gpt-5.6-terra"}  # same model as the pricing passes; see c2f/llm.py
TIMEOUT_S = 18.0  # needs 11-15 s on a 40-65k-char policy

SYSTEM = """You are reading a German insurance policy for a claims expert who will decide,
line by line, whether an invoice is payable. Extract ONLY the parts that BIND that decision.
Quote the policy's own wording and clause labels - do not paraphrase amounts, and never invent
a limit that is not in the text.

Answer with ONLY a JSON object of this exact shape and nothing else (no markdown fences):

{
  "insured_event": "what event is insured, in one or two sentences",
  "conditions": ["conditions of cover the claim must meet"],
  "limits": ["every sum insured, cap, sub-limit or valuation rule, with its amount and clause"],
  "deductibles": ["every deductible / excess, with its amount and clause"],
  "exclusions": ["every exclusion, with its clause"],
  "obligations": ["duties whose breach reduces or voids payment"]
}
Be exhaustive on limits, deductibles and exclusions - those are the point. Keep each entry
under 25 words. Use [] for a section the policy does not have."""


def render(d: dict) -> str:
    """Digest JSON -> the compact text block that goes into the prompt."""
    parts = []
    event = str(d.get("insured_event") or "").strip()
    if event:
        parts.append(f"INSURED EVENT: {event}")
    for key, label in (
        ("conditions", "CONDITIONS OF COVER"),
        ("limits", "LIMITS / SUMS INSURED"),
        ("deductibles", "DEDUCTIBLES"),
        ("exclusions", "EXCLUSIONS"),
        ("obligations", "OBLIGATIONS"),
    ):
        rows = [str(x).strip() for x in (d.get(key) or []) if str(x).strip()]
        if rows:
            parts.append(label + ":\n" + "\n".join(f"- {r}" for r in rows))
    return "\n\n".join(parts)


def _min_effort(model: str) -> str:
    """Lowest reasoning setting the model accepts: gpt-5.x calls it "none", gpt-5 "minimal"."""
    return "minimal" if model.startswith(("gpt-5-", "gpt-5o")) or model == "gpt-5" else "none"


def _call(text: str, model: str, timeout: float) -> str:
    user = f"<policy>\n{text}\n</policy>\n\nReturn the JSON now."
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        max_completion_tokens=4000,
        reasoning_effort=os.environ.get("C2F_REASONING_FAST", _min_effort(model)),
    )
    return resp.choices[0].message.content or ""


def distill(policy_text: str, *, timeout: float = TIMEOUT_S, model: str | None = None) -> tuple[str, dict]:
    """Full policy text -> (rendered digest, meta). Raises on failure; build() swallows."""
    prov = provider()
    model = model or os.environ.get("C2F_DIGEST_MODEL") or FAST[prov]
    t0 = time.time()
    raw = _call(policy_text, model, timeout)
    out = _parse_json(raw)
    text = render(out)
    if not text:
        raise ValueError("digest is empty")
    return text, {"model": model, "seconds": round(time.time() - t0, 2), "chars": len(text)}


def build(case_dir, *, timeout: float = TIMEOUT_S) -> tuple[str | None, dict]:
    """policy.txt -> (digest or None, meta). Never raises: the verbatim policy is enough.

    Returns rather than mutates so the caller can assign on its own thread - run.py runs this
    concurrently with the fast pass and must not race a model call that is already building
    its prompt from `case`.
    """
    src = Path(case_dir) / "policy.txt"
    if not src.exists():
        return None, {"skipped": "no policy.txt"}
    try:
        text, meta = distill(src.read_text(errors="replace"), timeout=timeout)
    except Exception as e:  # noqa: BLE001 - best-effort by design, see the module docstring
        return None, {"error": str(e)}
    return text, meta


def attach(case: dict, case_dir, *, timeout: float = TIMEOUT_S) -> dict:
    """build() + assign, for callers with no concurrency of their own (backtest, manual)."""
    text, meta = build(case_dir, timeout=timeout)
    if text:
        case["policy_digest"] = text
    return meta
