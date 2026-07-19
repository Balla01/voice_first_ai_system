"""
Per-PDF identity: doc_id, plan_name, doc_version, effective_date — plus the
"skip if this exact doc_id+version is already indexed" check that makes
re-running the pipeline re-embed only new/changed documents instead of the
whole corpus.

All of this is inferred from the filename (best-effort, same spirit as
lic_metadata.py's identify_doc_type — these are real-world LIC filenames,
not a clean schema), with file-mtime fallbacks where the filename doesn't
carry a date. Nothing here is authoritative; it exists so the payload has
*something* stable to filter/dedupe on rather than nothing.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# 'V02', 'V03', 'V01' style version markers seen in these filenames
# (e.g. "RiderEndorsement_AB Rider-V03-01102024-withlogo.pdf").
_VERSION_RE = re.compile(r"(?<![a-zA-Z0-9])V(\d{2,3})(?![a-zA-Z0-9])", re.IGNORECASE)

# ddmmyyyy (8 digits) then ddmmyy (6 digits) — try the longer pattern first so
# an 8-digit date isn't mistaken for a 6-digit one plus 2 stray digits.
_DATE_8_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)")
_DATE_6_RE = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")

_MONTH_NAME_RE = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*'?(\d{2,4})",
    re.IGNORECASE,
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)}

# Copy-suffixes ("(1)", "(2)") and doc-type/noise tokens to strip before
# deriving plan_name, so e.g. "745 LIC_Jeevan Umang_Eng _141025 (1).pdf" and
# "745 LIC_Jeevan Umang_Eng _141025.pdf" collapse to the same plan_name/doc_id
# (they're the same document, one just has an accidental copy suffix).
_COPY_SUFFIX_RE = re.compile(r"\(\d+\)")
_NOISE_TOKENS_RE = re.compile(
    r"(?<![a-zA-Z0-9])("
    r"CIS|Customer\s*Information\s*Sheet|Policy\s*Doc(?:ument)?|Pol\s*Doc|"
    r"Sales\s*Brochure|Brochure|Brouchure|Leaflet|Modif|Final|website|withlogo|"
    r"LIC'?s?|Eng|CC|V\d{2,3}"
    r")(?![a-zA-Z0-9])",
    re.IGNORECASE,
)


def _parse_filename_date(stem: str) -> Optional[str]:
    """Best-effort date extraction from a filename stem -> ISO 'YYYY-MM-DD' or None."""
    m = _DATE_8_RE.search(stem)
    if m:
        dd, mm, yyyy = m.groups()
        try:
            return datetime(int(yyyy), int(mm), int(dd)).date().isoformat()
        except ValueError:
            pass

    m = _DATE_6_RE.search(stem)
    if m:
        dd, mm, yy = m.groups()
        try:
            return datetime(2000 + int(yy), int(mm), int(dd)).date().isoformat()
        except ValueError:
            pass

    m = _MONTH_NAME_RE.search(stem)
    if m:
        month_str, year_str = m.groups()
        month = _MONTHS.get(month_str.lower()[:3])
        year = int(year_str)
        if year < 100:
            year += 2000
        if month:
            try:
                return datetime(year, month, 1).date().isoformat()
            except ValueError:
                pass

    return None


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return text.lower()


def derive_doc_metadata(pdf_path: Path, product_type: str, doc_type: str) -> dict:
    """
    Returns {"doc_id", "plan_name", "doc_version", "effective_date", "effective_date_source"}.

    doc_id is scoped to product_type + doc_type + plan_name (not plan_name
    alone) — a CIS and a Policy Document for the same plan are different
    documents and must not collapse into one doc_id.
    """
    stem = pdf_path.stem

    version_match = _VERSION_RE.search(stem)
    doc_version = f"V{version_match.group(1)}" if version_match else None

    effective_date = _parse_filename_date(stem)
    effective_date_source = "filename"
    if effective_date is None:
        effective_date = datetime.fromtimestamp(pdf_path.stat().st_mtime).date().isoformat()
        effective_date_source = "file_mtime"

    cleaned = _COPY_SUFFIX_RE.sub("", stem)
    cleaned = _DATE_8_RE.sub(" ", cleaned)
    cleaned = _DATE_6_RE.sub(" ", cleaned)
    cleaned = _NOISE_TOKENS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"^\s*\d+\s*", " ", cleaned)      # leading plan-number token, e.g. "717 "
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    plan_name = re.sub(r"\s+", " ", cleaned).strip(" _-") or stem

    doc_id = _slugify(f"{product_type}_{doc_type}_{plan_name}")

    return {
        "doc_id": doc_id,
        "plan_name": plan_name,
        "doc_version": doc_version or "unversioned",
        "effective_date": effective_date,
        "effective_date_source": effective_date_source,
    }


def already_indexed(client: QdrantClient, collection: str, doc_id: str, doc_version: str) -> bool:
    """True if any point for this exact doc_id+doc_version is already in the collection."""
    try:
        count = client.count(
            collection_name=collection,
            count_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="doc_version", match=MatchValue(value=doc_version)),
            ]),
        ).count
        return count > 0
    except Exception:
        # Collection may not exist yet on the very first run.
        return False
