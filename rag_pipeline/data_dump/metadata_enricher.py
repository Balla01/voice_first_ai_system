"""
LLM-based per-chunk metadata tagging, using the local Qwen3.5 model
(adapted from miscellaneous/qwen_test.py — see that file for the model-card
generation settings this mirrors).

Gated by constants.USE_LLM_METADATA: this box has no CUDA, so generation runs
on CPU. The model is loaded lazily on first use so a pipeline run with the
flag off never pays the ~4-5GB download/load cost.
"""
import json
import re
import threading
from typing import Optional

from constants import QWEN_METADATA_MODEL, QWEN_METADATA_MAX_NEW_TOKENS

SYSTEM_PROMPT = """You are a metadata extraction assistant for insurance policy documents.
Given a chunk of text from an LIC policy document, extract structured metadata and
respond with ONLY a valid JSON object — no explanation, no markdown fences, no extra text.

The JSON must have exactly these fields:
{
  "section_title": string,
  "chunk_type": one of ["definition", "clause", "table", "benefit_description", "eligibility", "exclusion", "other"],
  "key_terms": array of strings,
  "contains_table": boolean,
  "clause_numbers": array of strings,
  "summary": string (one sentence)
}

Only use information present in the text. Do not infer or add information not stated."""

_model = None
_processor = None
_device = None
_lock = threading.Lock()


def _get_model():
    global _model, _processor, _device
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                _device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading metadata LLM: {QWEN_METADATA_MODEL} (device={_device})")
                _processor = AutoProcessor.from_pretrained(QWEN_METADATA_MODEL)
                _model = AutoModelForImageTextToText.from_pretrained(
                    QWEN_METADATA_MODEL,
                    torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
                    device_map=_device,
                )
                _model.eval()
                print("Metadata LLM ready.")
    return _model, _processor, _device


def _parse_json(raw_text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{raw_text}")
    return json.loads(match.group(0))


_EMPTY_RESULT = {
    "section_title": "",
    "chunk_type": "other",
    "key_terms": [],
    "contains_table": False,
    "clause_numbers": [],
    "summary": "",
}


def extract_metadata(chunk_text: str, max_new_tokens: int = QWEN_METADATA_MAX_NEW_TOKENS) -> dict:
    """
    Returns the metadata dict, or _EMPTY_RESULT (with an "extraction_error" key
    added) on any generation/parse failure — callers should never have to
    special-case a missing field.
    """
    import torch

    model, processor, device = _get_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract metadata from this chunk:\n\n{chunk_text}"},
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                top_k=20,
                do_sample=False,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        print('META DATA', raw_text)
        return _parse_json(raw_text)
    except Exception as e:
        result = dict(_EMPTY_RESULT)
        result["extraction_error"] = str(e)
        return result
