"""One model call per case. OpenAI only (default model gpt-5.6-sol, override C2F_MODEL).

The model sees policy + description + invoice verbatim and returns JSON.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time

from c2f.env import load_dotenv

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
              Use market prices in Germany in 2026 for a standard / mid-range
              replacement or a typical tradesman rate - NOT premium brands, NOT
              worst-case. Claims experts are frugal: when unsure, go lower. Use any
              value hints in the damage description (stated worth, age, make, size). Respect policy limits
              (sum insured, market value, caps, deductibles) - t must not exceed them.

              BUT the description's own characterisation of the item binds: if it
              calls something "expensive", "designer", "luxury", "premium",
              "high-end" or names a premium brand, the fair value is
              that tier's like-for-like replacement (an "expensive watch" is a
              several-thousand-euro watch, not a standard one). Frugal means no
              invented upgrades, not ignoring stated value.

              NEVER INVENT SPECIFICS. If the invoice/description does not state a brand,
              model, screen size, wattage, channel count or other premium feature, do not
              assume one - price the CHEAPEST reasonable standard replacement that matches
              only what is actually stated, not a mid-range guess dressed up with invented
              detail. When the evidence for a number is weak (no stated size/spec/age/price),
              make t_low strongly conservative - closer to the cheapest plausible item than
              to t_mid - because a low t_low costs nothing (it only ever raises b) while an
              inflated one risks accepting fraud.

              Bundled diagnostic, inspection, call-out or service charges: price only the
              ONE necessary visit/report, never the full billed quantity, unless the
              invoice or description gives a specific reason for more than one.

   If the policy text refers to a cap, sub-limit or schedule value for this item's category
   but does not state the number (e.g. "as per Appendix B" with no figure given), set
   "cap_uncertain": true and keep your estimate conservative - do not guess the cap's value.
   Otherwise omit cap_uncertain or set it false.

   If the item is not covered OR not related, set t_low = t_mid = t_high = 0 and put
   your estimate of what the item WOULD cost if it were payable into t_if_covered.

   This zero convention is ONLY for not-covered/not-related items. A covered AND related
   item must always get t_mid > 0 - even when the exact cap/sub-limit figure is unknown
   (cap_uncertain: true), still give your best frugal market-value estimate for the item
   itself. Never leave t_low = t_mid = t_high = 0 for an item you marked covered.

   If covered=true and related=true, t_mid and t_high MUST be positive. Uncertainty, an
   unstated cap, missing specifications, or an upgrade is not permission to return zero.
   Estimate the cheapest reasonable like-for-like value and express uncertainty through
   t_low/t_high and cap_uncertain.

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
      "cap_uncertain": false,
      "reason": "max 12 words"
    }
  ]
}
Every line item index that appears on the invoice MUST appear exactly once."""


def build_user_message(case: dict, only: list[int] | None = None, sweep: bool = False) -> str:
    items_txt = ""
    if case.get("items"):
        rows = "\n".join(
            f"  {it['index']} | {it['description']} | {it['quantity']:g} {it['unit']}" for it in case["items"]
        )
        items_txt = (
            "\n<parsed_line_items note=\"a mechanical parse of the invoice above, as a reading aid; it can be "
            "INCOMPLETE - every POS. number that appears in the invoice text must be priced, whether or not it is "
            "listed here\">\n" + rows + "\n</parsed_line_items>\n"
        )
    meta = case.get("invoice_meta") or {}
    meta_txt = " ".join(f'{k}="{v}"' for k, v in meta.items())
    # A fast model's extract of the clauses that bind (c2f.policy). Absent if that call failed:
    # it is a reading aid in front of the verbatim policy below, never a replacement for it.
    digest = case.get("policy_digest")
    digest_txt = (
        f"<policy_digest note=\"limits, deductibles and exclusions pulled out of the policy below; "
        f"a reading aid - the full policy text is authoritative\">\n{digest}\n</policy_digest>\n\n"
        if digest
        else ""
    )
    return (
        digest_txt
        + f"<policy>\n{case['policy']}\n</policy>\n\n"
        f"<damage_description>\n{case['description']}\n</damage_description>\n\n"
        f"<invoice {meta_txt}>\n{case['invoice_text']}\n</invoice>\n"
        f"{items_txt}"
        + ("\nImages from the case are attached.\n" if case.get("images") else "")
        # Chunking keeps the WHOLE policy and invoice in front of the model and narrows only
        # the answer. Splitting the context instead would cost the cross-item signal - a
        # duplicated line, or labour inconsistent with the parts fitted - which is exactly
        # what a single whole-invoice call was for.
        + (
            "\nPRICE ONLY THESE POS NUMBERS: "
            + ", ".join(str(i) for i in only)
            + f". Return exactly {len(only)} objects, one per listed POS number"
            + (
                # One chunk keeps the job of catching what the parser missed. Game 11 had 22
                # line items across several invoices, the parser found 11, and the 11 it never
                # saw went out at 0/0 - two thirds of that round's loss.
                ", PLUS one object for every POS number that appears in the invoice text above"
                " but is missing from the parsed list. Do not lose those.\n"
                if sweep
                else ", and no others. The rest of the invoice is context only.\n"
            )
            if only
            else ""
        )
        + "\nReturn the JSON now."
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def _call_openai(case: dict, model: str, timeout: float, system: str = SYSTEM, fast: bool = False, only: list[int] | None = None, sweep: bool = False) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=0)
    content: list[dict] = [{"type": "text", "text": build_user_message(case, only, sweep)}]
    # Only the sweep chunk (or an unchunked call) carries the images. Every chunk sees the
    # same case, so re-uploading them per chunk multiplies the cost that misses deadlines.
    for img in (case.get("images", []) if (only is None or sweep) else []):
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{img['media_type']};base64,{img['b64']}"}}
        )
    kwargs: dict = {}
    if model.startswith(("gpt-5", "o")):
        # The fast pass (and any mini/nano model) runs at the effort floor: it is the safety
        # submission and must land early - gpt-5.6-terra at "low" took 35 s on a 32-item case,
        # at "none" 29 s; on small cases 9 s vs 5 s. gpt-5.x rejects "minimal"; its floor is "none".
        small = fast or "mini" in model or "nano" in model
        default = ("minimal" if model.startswith("gpt-5-") else "none") if small else "low"
        kwargs["reasoning_effort"] = os.environ.get("C2F_REASONING_FAST" if small else "C2F_REASONING", default)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        max_completion_tokens=8000,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def provider() -> str:
    """OpenAI only. Raises if no key is configured."""
    load_dotenv()
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError("no LLM key: set OPENAI_API_KEY (env or .env)")


STRICT_SUFFIX = "\n\nIMPORTANT: you are the quick first pass. When in doubt whether an item is covered or related, answer covered=false (we can be corrected later, but a wrong acceptance pays a fraud)."


ADDENDUM_PATH = pathlib.Path(__file__).resolve().parent.parent / "runs" / "prompt_addendum.txt"


def addendum() -> str:
    """Extra prompt rules under test, appended to SYSTEM when the file exists.

    `c2f.propose` writes a candidate rule here so `make replay` can score it
    against past rounds before anyone edits SYSTEM. Delete the file to revert.
    """
    try:
        text = ADDENDUM_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return f"\n\n{text}" if text else ""


def estimate(case: dict, *, timeout: float = 35.0, model: str | None = None, strict: bool = False,
             only: list[int] | None = None, sweep: bool = False) -> tuple[dict, dict]:
    """Return (model_json, meta). Raises on failure; caller decides the fallback.

    `only` restricts the ANSWER to those POS numbers while leaving the whole case in
    the prompt. c2f.run uses it to split a large invoice into short parallel calls so a
    slow one costs its own items rather than the entire round."""
    from c2f.validate import validate_items  # local import: keeps price/validate free of llm's provider deps

    prov = provider()
    t0 = time.time()
    system = SYSTEM + (STRICT_SUFFIX if strict else "") + addendum()
    model = model or os.environ.get("C2F_MODEL") or "gpt-5.6-sol"
    raw = _call_openai(case, model, timeout, system, fast=strict, only=only, sweep=sweep)
    out = _parse_json(raw)
    if "items" not in out or not isinstance(out["items"], list):
        raise ValueError("model JSON has no items list")
    meta = {"provider": prov, "model": model, "seconds": round(time.time() - t0, 2), "raw": raw}
    errors = validate_items(out)
    if errors:
        meta["validation_errors"] = [str(e) for e in errors]
    return out, meta
