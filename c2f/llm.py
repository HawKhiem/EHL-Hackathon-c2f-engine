"""One model call per case. Provider picked by which key is set.

ANTHROPIC_API_KEY -> Claude (default model claude-opus-5, override C2F_MODEL)
OPENAI_API_KEY    -> OpenAI (default model gpt-5, override C2F_MODEL)
neither / --mock  -> canned answer (for testing the pipeline)

The model sees policy + description + invoice verbatim and returns JSON.
"""

from __future__ import annotations

import json
import os
import re
import time

SYSTEM = """You are a senior insurance claims expert working in Germany. You assess invoices
submitted after an insured event. For EVERY line item on the invoice you decide:

1. covered  - Is this type of cost insured under the policy text? Check the insured
              event, conditions of cover, amount of indemnity and exclusions.
2. related  - Does this line item belong to the reported damage described? An item
              that is unrelated to the event (or clearly excessive in kind, e.g. an
              upgrade instead of a repair) is not related.
3. fair gross total - the TOTAL price for the whole line (quantity x unit price,
              INCLUDING 19% German VAT, in EUR) that a careful claims expert would
              still accept as reasonable. Give three numbers:
                t_low  : a price almost certainly acceptable
                t_mid  : your best single estimate of the maximum acceptable price
                t_high : a price that is probably already too expensive
              Use market prices in Germany in 2026. Use any value hints in the damage
              description (stated worth, age, make, size). Respect policy limits
              (sum insured, market value, caps, deductibles) - t must not exceed them.
   If the item is not covered OR not related, set t_low = t_mid = t_high = 0 and put
   your estimate of what the item WOULD cost if it were payable into t_if_covered.

Be concrete, decisive and FAST (you have 30 seconds). Do not deliberate; answer with ONLY a JSON object of this
exact shape and nothing else (no markdown fences):

{
  "policy_summary": "one line: what is insured, main limits/exclusions",
  "items": [
    {
      "index": 1,
      "covered": true,
      "related": true,
      "clause": "policy clause relied on, e.g. '4. Amount of indemnity'",
      "t_low": 380,
      "t_mid": 420,
      "t_high": 450,
      "t_if_covered": 0,
      "reason": "max 12 words"
    }
  ]
}
Every line item index that appears on the invoice MUST appear exactly once."""


def build_user_message(case: dict) -> str:
    items_txt = ""
    if case.get("items"):
        rows = "\n".join(
            f"  {it['index']} | {it['description']} | {it['quantity']:g} {it['unit']}" for it in case["items"]
        )
        items_txt = f"\n<parsed_line_items>\n{rows}\n</parsed_line_items>\n"
    meta = case.get("invoice_meta") or {}
    meta_txt = " ".join(f'{k}="{v}"' for k, v in meta.items())
    return (
        f"<policy>\n{case['policy']}\n</policy>\n\n"
        f"<damage_description>\n{case['description']}\n</damage_description>\n\n"
        f"<invoice {meta_txt}>\n{case['invoice_text']}\n</invoice>\n"
        f"{items_txt}"
        + ("\nImages from the case are attached.\n" if case.get("images") else "")
        + "\nReturn the JSON now."
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def _mock(case: dict) -> dict:
    items = case.get("items") or [{"index": 1, "description": "?"}]
    return {
        "policy_summary": "MOCK",
        "items": [
            {
                "index": it["index"],
                "description": it.get("description", ""),
                "covered": True,
                "related": True,
                "clause": "mock",
                "t_low": 380,
                "t_mid": 420,
                "t_high": 450,
                "t_if_covered": 0,
                "reason": "mock answer",
            }
            for it in items
        ],
    }


def _call_anthropic(case: dict, model: str, timeout: float, system: str = SYSTEM) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=timeout, max_retries=0)
    content: list[dict] = [{"type": "text", "text": build_user_message(case)}]
    for img in case.get("images", []):
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": img["media_type"], "data": img["b64"]},
            }
        )
    kwargs: dict = {}
    if model.startswith("claude-opus-5") or model.startswith("claude-fable") or model.startswith("claude-sonnet-5"):
        kwargs["thinking"] = {"type": "adaptive"}
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": content}],
        **kwargs,
    )
    if getattr(resp, "stop_reason", None) == "refusal":
        raise RuntimeError("model refused")
    return "".join(getattr(b, "text", "") for b in resp.content)


def _call_openai(case: dict, model: str, timeout: float, system: str = SYSTEM) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=0)
    content: list[dict] = [{"type": "text", "text": build_user_message(case)}]
    for img in case.get("images", []):
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{img['media_type']};base64,{img['b64']}"}}
        )
    kwargs: dict = {}
    if model.startswith(("gpt-5", "o")):
        default = "minimal" if "mini" in model or "nano" in model else "low"
        kwargs["reasoning_effort"] = os.environ.get("C2F_REASONING_FAST" if default == "minimal" else "C2F_REASONING", default)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        max_completion_tokens=8000,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def provider() -> str:
    if os.environ.get("C2F_MOCK"):
        return "mock"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "mock"


STRICT_SUFFIX = "\n\nIMPORTANT: you are the quick first pass. When in doubt whether an item is covered or related, answer covered=false (we can be corrected later, but a wrong acceptance pays a fraud)."


def estimate(case: dict, *, timeout: float = 35.0, mock: bool = False, model: str | None = None, strict: bool = False) -> tuple[dict, dict]:
    """Return (model_json, meta). Raises on failure; caller decides the fallback."""
    prov = "mock" if mock else provider()
    t0 = time.time()
    system = SYSTEM + (STRICT_SUFFIX if strict else "")
    if prov == "anthropic":
        model = model or os.environ.get("C2F_MODEL") or "claude-opus-5"
        raw = _call_anthropic(case, model, timeout, system)
    elif prov == "openai":
        model = model or os.environ.get("C2F_MODEL") or "gpt-5"
        raw = _call_openai(case, model, timeout, system)
    else:
        model = "mock"
        raw = json.dumps(_mock(case))
    out = _parse_json(raw)
    if "items" not in out or not isinstance(out["items"], list):
        raise ValueError("model JSON has no items list")
    return out, {"provider": prov, "model": model, "seconds": round(time.time() - t0, 2), "raw": raw}
