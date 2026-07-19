"""
Dynamic customer-profile extraction.

On each customer turn we ask the shared chat LLM (the same one Layer 3 uses via
get_llm_client() — an API model, NOT the local embedding model that needs GPU/
RAM) to pull structured facts out of the running transcript. The result is
merged into a per-session profile dict that the UI renders in a single card that
fills in as the call goes on:

    name -> then profession + location -> then family, all in the same card.

LLM-only by design (per product decision): open-ended fields like profession and
family phrasing are brittle to regex. We keep the call cheap (small max_tokens,
strict JSON contract) and run it async so it never blocks STT/routing.
"""

import json
import logging

logger = logging.getLogger("insureassist.profile")

PROFILE_FIELDS = ("name", "age", "profession", "location", "family")

_PROMPT = """You extract a customer's profile from a live insurance sales call.

Below is the recent transcript. Return ONLY a JSON object with these keys:
  "name":       full name if the CUSTOMER states it, else null
  "age":        integer age if stated, else null
  "profession": job/occupation if stated, else null
  "location":   city/area/state if stated, else null
  "family":     list of short strings for family members mentioned
                (e.g. ["wife (diabetic)", "2 children", "father-in-law, 68"]),
                or [] if none

Rules:
- Only include facts the CUSTOMER explicitly stated. Do not guess or infer.
- If a field is not mentioned, use null (or [] for family).
- Output raw JSON only. No prose, no markdown fences.

Transcript:
{transcript}
"""


def _coerce(raw: str) -> dict | None:
    """Pull a JSON object out of the model's reply, tolerating code fences."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        # drop an optional leading "json" language tag
        if s[:4].lower() == "json":
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        logger.debug("profile: could not parse LLM JSON: %r", raw)
        return None


async def extract_profile(llm, transcript_text: str) -> dict | None:
    """Return {name, age, profession, location, family} (fields may be null/[]),
    or None if extraction failed. Never raises — profile is best-effort."""
    if not transcript_text.strip():
        return None
    try:
        reply = await llm.create_message(
            max_tokens=200, prompt=_PROMPT.format(transcript=transcript_text[-4000:])
        )
    except Exception as e:  # noqa: BLE001 — LLM is optional, don't break the call
        logger.warning("profile extraction LLM call failed: %s", e)
        return None

    data = _coerce(reply)
    if not isinstance(data, dict):
        return None

    out = {k: data.get(k) for k in PROFILE_FIELDS}
    fam = out.get("family")
    out["family"] = [str(x).strip() for x in fam if x] if isinstance(fam, list) else []
    return out


def merge_profile(current: dict, incoming: dict) -> bool:
    """Merge newly extracted facts into the session profile in place.
    Scalar fields overwrite when the new value is non-empty; family members are
    appended + de-duplicated. Returns True if anything changed (so the caller
    only pushes to the UI on real updates)."""
    changed = False
    for f in ("name", "age", "profession", "location"):
        v = incoming.get(f)
        if v not in (None, "", []) and current.get(f) != v:
            current[f] = v
            changed = True
    for m in incoming.get("family", []) or []:
        if m and m not in current.setdefault("family", []):
            current["family"].append(m)
            changed = True
    return changed


def new_profile() -> dict:
    return {"name": None, "age": None, "profession": None, "location": None, "family": []}
