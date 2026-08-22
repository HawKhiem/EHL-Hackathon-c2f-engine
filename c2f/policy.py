"""Structured pre-extraction of a policy that is too long to send verbatim.

Only policy.txt runs long (40-65k chars in the games so far). Capping it at
extract.MAX_CHARS drops the tail - and the tail is where the exclusions, caps and
deductibles live, i.e. exactly what decides `covered`. So when a field was actually cut,
we ask the FASTEST model for a structured digest of the WHOLE file and put that in front
of the capped verbatim text. Nothing that binds gets silently dropped.

This runs only when extract reported a cut (`case["truncated"]`), so on a normal case it
costs nothing - which matters, the game has a 60 s clock. It is best-effort by design:
any failure or timeout leaves the case untouched and the run proceeds on capped text.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from c2f.llm import _parse_json, provider

# Fastest model per provider. Override with C2F_DIGEST_MODEL.
FAST = {"anthropic": "claude-haiku-4-5", "openai": "gpt-5-nano"}
TIMEOUT_S = 12.0

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


def _call(text: str, model: str, prov: str, timeout: float) -> str:
    user = f"<policy>\n{text}\n</policy>\n\nReturn the JSON now."
    if prov == "anthropic":
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"], timeout=timeout, max_retries=0
        )
        resp = client.messages.create(
            model=model,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            raise RuntimeError("model refused")
        return "".join(getattr(b, "text", "") for b in resp.content)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        max_completion_tokens=4000,
        reasoning_effort=os.environ.get("C2F_REASONING_FAST", "minimal"),
    )
    return resp.choices[0].message.content or ""


def distill(policy_text: str, *, timeout: float = TIMEOUT_S, model: str | None = None) -> tuple[str, dict]:
    """Full policy text -> (rendered digest, meta). Raises on failure; attach() swallows."""
    prov = provider()
    model = model or os.environ.get("C2F_DIGEST_MODEL") or FAST[prov]
    t0 = time.time()
    raw = _call(policy_text, model, prov, timeout)
    out = _parse_json(raw)
    text = render(out)
    if not text:
        raise ValueError("digest is empty")
    return text, {"model": model, "seconds": round(time.time() - t0, 2), "chars": len(text)}


def attach(case: dict, case_dir, *, timeout: float = TIMEOUT_S) -> dict | None:
    """Set case["policy_digest"] iff the policy was cut. Never raises - returns meta or None."""
    if "policy" not in (case.get("truncated") or []):
        return None
    src = Path(case_dir) / "policy.txt"
    if not src.exists():
        return None
    try:
        text, meta = distill(src.read_text(errors="replace"), timeout=timeout)
    except Exception as e:  # noqa: BLE001 - best-effort: capped text is still a valid prompt
        return {"error": str(e)}
    case["policy_digest"] = text
    return meta
