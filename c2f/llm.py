"""One model call per case. OpenAI only (default model gpt-5.6-terra, override C2F_MODEL).

The model sees policy + description + invoice verbatim and returns JSON.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time

from c2f.env import load_dotenv
from c2f.history import HISTORY

SYSTEM = """You are a senior insurance claims expert working in Germany. You assess invoices
submitted after an insured event. For EVERY line item on the invoice you decide:

1. covered  - Is this type of cost insured under the policy text? Check the insured
              event, conditions of cover, amount of indemnity and exclusions. When in doubt,
              prefer COVERED unless there is explicit evidence the item is excluded. Only answer
              NOT covered when:
              - the policy explicitly names the item/category in an exclusions clause, OR
              - the cost type is clearly unrelated to insured events (e.g. upgrades, preventive
                maintenance, unrelated damage, cosmetic-only repairs), OR
              - the item predates the insured event or is for scheduled maintenance.
              Ambiguity about quantity, condition, or price does NOT make an item uncovered.
              Uncertainty about the exact clause that covers it is NOT a reason to say NOT covered.
2. related  - Does this line item belong to the reported damage described? An item
              that is unrelated to the event (or clearly excessive in kind, e.g. an
              upgrade instead of a repair) is not related. When unsure if an item relates to
              the described damage, prefer RELATED unless there is clear evidence it does not.
3. fair gross total - the TOTAL price for the whole line (quantity x unit price,
              INCLUDING 19% German VAT, in EUR) that a careful claims expert would
              still accept as reasonable. Break this down into:
                - What is the unit of the cost? (per hour, per item, per m², flat call-out, project price)
                - How many units are involved? (from the quantity field, or reason from the description)
                - What is a plausible market unit price in Germany in 2026?
                - Multiply to get the gross total including VAT.
              Then give three numbers - honest quantiles of your belief about the threshold,
              not negotiating positions:
                t_low  : 10th percentile - only a 10% chance the true threshold is below this
                t_mid  : your median estimate of the threshold (the maximum acceptable price)
                t_high : 90th percentile - only a 10% chance the true threshold is above this
              Use market prices in Germany in 2026 for a standard / mid-range
              replacement or a typical tradesman rate - NOT premium brands, NOT
              worst-case. The threshold applies to the line item AS INVOICED: keep the invoiced
              quantity for materials, parts, hours and rental days - reviewers accept the full
              line unless the quantity itself is clearly unjustified for the described damage.
              Past rounds show reviewers accept normal German market gross rates, not
              bargain-basement prices - aim for the realistic market price, neither padded nor
              undercut. Use any value hints in the damage description (stated worth, age, make,
              size). Respect policy limits (sum insured, market value, caps, deductibles) -
              t must not exceed them.

              BUT the description's own characterisation of the item binds: if it
              calls something "expensive", "designer", "luxury", "premium",
              "high-end" or names a premium brand, the fair value is
              that tier's like-for-like replacement (an "expensive watch" is a
              several-thousand-euro watch, not a standard one). Frugal means no
              invented upgrades, not ignoring stated value.

              NEVER INVENT SPECIFICS. If the invoice/description does not state a brand,
              model, screen size, wattage, channel count or other premium feature, do not
              assume one - price a standard mid-market replacement that matches only what is
              actually stated. When the evidence for a number is weak (no stated
              size/spec/age/price), express that as a WIDE t_low..t_high interval, not by
              dragging t_mid down.

              Repeated call-out, travel or inspection FEES (the same flat fee billed several
              times): allow only the visits the described damage actually needed, unless the
              invoice or description justifies more. This applies to duplicated service fees
              only - never cut the quantity of materials, parts, labour hours or rental days
              below what the invoice states.

              MARKET EVIDENCE BINDS: when <market_history> lists a line item verbatim or a
              matching category, your t_mid must not sit below its proven floor without a
              case-specific reason (a policy cap, a stated lower value). Past reviewers
              accepted those prices; undercutting them has cost real money every round.

              DECLARED / SCHEDULED ITEMS: when the description or policy says an item is
              individually declared on a valuables schedule, has a valuation certificate, or
              is insured at an agreed value, the certificate/agreed value governs t - NOT a
              generic replacement estimate. "Declared at a value well above the standard
              per-item limit" means exactly that: take the standard sub-limit as a FLOOR,
              set t_mid well above it (past rounds settled such items at EUR 7,000-11,000+),
              and carry the upside in t_high. Under-pricing a scheduled valuable has been
              one of this system's largest single-round losses, twice.

              SPECIALIST-TIER SERVICES: when the case involves a high-value or specialist
              object (fine art, a painting, antiques, jewellery, instruments), EVERY service
              touching that object - assessment, transport, stabilisation, conservation,
              restoration - is priced on the SPECIALIST market (accredited conservators, art
              handlers), typically 2-4x an ordinary tradesman's rate, and it scales with the
              object's stated value tier. A "full restoration" of a high-value painting is a
              conservator's project (thousands of euros), not a redecorating job. Pricing
              fine-art services like ordinary trades has been this system's largest error.

   COMMON FAILURE MODES TO AVOID:
   - Confusing repair/restoration cost with replacement cost: repair is always cheaper.
   - Pricing the whole project when the invoice is for one small component within it.
   - Misunderstanding quantity: if quantity=1 but description says "per hour" or "per m²",
     check if the invoice text specifies the actual hours/area - the quantity field is often
     generic, and the fair total covers ALL invoiced units (hours x rate, m² x rate, days x
     rate), not one unit.
   - Under-pricing small material and consumable lines: even minor parts carry handling,
     gross margin and VAT - a line's fair total is rarely below the trade's minimum charge.
   - Pricing a RENTAL line ("unit rental", "hire") at the equipment's purchase price: a
     rental is the daily/weekly rate x the billed period stated in the invoice - usually a
     small fraction of buying the machine.
   - Assuming "expensive" items are luxury goods: the invoice context (e.g. "replaced the damaged
     watch") means fair market replacement of that stated tier, not premium resale value.

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
    # No <parsed_line_items> block: the invoice is not line-parsed (see c2f.extract). The
    # model reads the full pdf text and decides for itself what the line items are.
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
        # Past rounds' recovered t brackets (c2f.history). Evidence for the size of a number,
        # in front of the case so it frames the reading rather than second-guessing it after.
        HISTORY
        + digest_txt
        + f"<policy>\n{case['policy']}\n</policy>\n\n"
        f"<damage_description>\n{case['description']}\n</damage_description>\n\n"
        f"<invoice {meta_txt}>\n{case['invoice_text']}\n</invoice>\n"
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
    # Text only: case photos are skipped (case["images"] is names, no bytes). They cost
    # upload time and tokens inside a 60 s window and the pricing decision is made from the
    # policy, the description and the invoice.
    content: list[dict] = [{"type": "text", "text": build_user_message(case, only, sweep)}]
    kwargs: dict = {}
    if model.startswith(("gpt-5", "o")):
        # The fast pass (and any mini/nano model) runs at the effort floor: it is the safety
        # submission and must land early - gpt-5.6-terra at "low" took 35 s on a 32-item case,
        # at "none" 29 s; on small cases 9 s vs 5 s. gpt-5.x rejects "minimal"; its floor is "none".
        small = fast or "mini" in model or "nano" in model
        # The full pass thinks harder now that it is the ONLY call: one model, one pass, and at
        # most CHUNK_ITEMS items per call, so the effort is spent on a short prompt-relative job.
        default = ("minimal" if model.startswith("gpt-5-") else "none") if small else "medium"
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

# ---- whale resampling: median-of-k on the items that decide the round -------------------
# One sample of t_mid on a dominant item swings the round: game 41's robbery compensation
# came out 8,000 on one run and 6,000 on the next (same prompt), a +-65k difference in net.
# For items whose stake crosses RESAMPLE_T we ask the model again (answer narrowed via
# `only`, whole case still in the prompt) and combine per item: majority coverage, median
# t values, and a between-sample sigma that widens the pricing belief exactly when the
# samples disagree.
RESAMPLE_T = float(os.environ.get("C2F_RESAMPLE_T") or 1000.0)
RESAMPLE_K = int(os.environ.get("C2F_RESAMPLE_K") or 2)  # extra samples (total = 1 + K)


def _stake(it: dict) -> float:
    for k in ("t_mid", "t_if_covered"):
        try:
            v = float(it.get(k) or 0)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return 0.0


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def resample_whales(case: dict, out: dict, *, k: int = RESAMPLE_K, threshold: float = RESAMPLE_T,
                    timeout: float = 25.0, model: str | None = None) -> dict:
    """Return `out` with every whale item replaced by the sample-combined version.

    Best effort: a failed resample leaves the original single-sample item untouched."""
    import math
    from concurrent.futures import ThreadPoolExecutor

    big = [int(it["index"]) for it in out.get("items", [])
           if str(it.get("index", "")).lstrip("-").isdigit() and _stake(it) >= threshold]
    if not big or k <= 0:
        return out
    samples: list[dict[int, dict]] = []
    with ThreadPoolExecutor(max_workers=k) as ex:
        futs = [ex.submit(estimate, dict(case), timeout=timeout, model=model, only=list(big)) for _ in range(k)]
        for f in futs:
            try:
                extra, _meta = f.result()
                samples.append({int(it["index"]): it for it in extra.get("items", [])
                                if str(it.get("index", "")).lstrip("-").isdigit()})
            except Exception:  # noqa: BLE001 - a failed resample must never sink the round
                continue
    if not samples:
        return out
    merged_items = []
    for it in out.get("items", []):
        try:
            idx = int(it["index"])
        except (KeyError, TypeError, ValueError):
            merged_items.append(it)
            continue
        if idx not in big:
            merged_items.append(it)
            continue
        pool = [it] + [s[idx] for s in samples if idx in s]
        # The MAIN pass's coverage call stands - a resample exists to stabilise the NUMBERS.
        # Letting a 2-of-3 vote flip coverage turned game 40's covered restoration (t proven
        # in [2137, 2880)) into a b=0 refusal of every fair charge. Only samples that agree
        # with the main coverage contribute medians; if none agree, the item stays as-is.
        covered = bool(it.get("covered", False)) and bool(it.get("related", True))
        agree = [p for p in pool if (bool(p.get("covered", False)) and bool(p.get("related", True))) == covered]
        combined = dict(it)
        for key in ("t_low", "t_mid", "t_high", "t_if_covered"):
            vals = [float(p.get(key) or 0) for p in agree if float(p.get(key) or 0) > 0]
            if vals:
                combined[key] = round(_median(vals), 2)
        mids = [float(p.get("t_mid") or 0) for p in pool if float(p.get("t_mid") or 0) > 0]
        if len(mids) >= 2:
            combined["_sample_sigma"] = round((math.log(max(mids)) - math.log(min(mids))) / 2, 4)
        combined["_n_samples"] = len(pool)
        merged_items.append(combined)
    return {**out, "items": merged_items}


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
    # One model for every pass. Paired on the 9 games where both ran on the same case and the
    # same prompt (12,13,14,16,17,19,20,21,22, scored against runs/truth_game_*.json), terra and
    # sol call coverage identically (89% each) and terra prices no worse (median |log t error|
    # 0.33 vs 0.39) - at roughly a third of the latency (2.4 s vs 6.8 s on game 22), which inside
    # a 60 s clock is the whole ballgame. No measured reason to pay for two.
    model = model or os.environ.get("C2F_MODEL") or "gpt-5.6-terra"
    raw = _call_openai(case, model, timeout, system, fast=strict, only=only, sweep=sweep)
    out = _parse_json(raw)
    if "items" not in out or not isinstance(out["items"], list):
        raise ValueError("model JSON has no items list")
    meta = {"provider": prov, "model": model, "seconds": round(time.time() - t0, 2), "raw": raw}
    errors = validate_items(out)
    if errors:
        meta["validation_errors"] = [str(e) for e in errors]
    return out, meta
