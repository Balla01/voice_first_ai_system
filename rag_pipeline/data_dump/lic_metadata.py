"""
Product-PDF folder walker + metadata extraction.

Layout handled (flat — no category/sub_category/plan-number nesting):
  PRODUCT_PDFS_ROOT/insurance-plan/*.pdf
  PRODUCT_PDFS_ROOT/pinsion_data/*.pdf

For each PDF this yields (pdf_path, metadata) where metadata is just
product_type (from PRODUCT_TYPE_FOLDERS) plus a doc_type identified from
the filename/title.
"""

import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pdfplumber

from constants import PRODUCT_TYPE_FOLDERS

# Note: '_' counts as a \w character in Python regex, so plain \b does not fire
# between a letter and '_' (e.g. 'CIS_Amritbaal'). These filenames use '_' as a
# word separator throughout, so boundaries are expressed as lookarounds instead.
_NOT_ALNUM_BEFORE = r'(?<![a-zA-Z0-9])'
_NOT_ALNUM_AFTER = r'(?![a-zA-Z0-9])'

_CIS_RE = re.compile(
    rf'{_NOT_ALNUM_BEFORE}cis{_NOT_ALNUM_AFTER}|customer\s*information\s*sheet',
    re.IGNORECASE,
)
# Matches 'Policy Document', 'Policy_Document', 'Policy Doc', 'Pol Document', etc.
_POLICY_DOC_RE = re.compile(r'pol(?:icy)?[\s_]*doc(?:ument)?', re.IGNORECASE)

# 'brouchure' is a common real-world misspelling in this dataset; 'leaflet' is
# LIC's own name for the same 4x9" promotional pamphlet format as its brochures.
_BROCHURE_RE = re.compile(
    rf'{_NOT_ALNUM_BEFORE}(brochure|brouchure|leaflet){_NOT_ALNUM_AFTER}',
    re.IGNORECASE,
)


def _extract_pdf_title_and_text(pdf_path: Path) -> Tuple[str, str]:
    """
    Best-effort (title, first_page_text).
    title = PDF metadata Title, else first meaningful line of page 1.
    Metadata Title is often export junk (e.g. an InDesign filename), so callers
    should classify against first_page_text too, not the title alone.
    """
    title = ""
    first_page_text = ""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if pdf.pages:
                first_page_text = pdf.pages[0].extract_text() or ""

            meta_title = ((pdf.metadata or {}).get("Title") or "").strip()
            if meta_title and meta_title.lower() not in ("untitled", "microsoft word"):
                title = meta_title
            else:
                for line in first_page_text.splitlines():
                    line = line.strip()
                    if len(line) >= 5:
                        title = line
                        break
    except Exception:
        pass
    return title, first_page_text


def identify_doc_type(pdf_path: Path) -> str:
    """
    Filename contains 'brochure' -> 'Sales Brochure'.
    Otherwise read the PDF's title (metadata or first-page heading) plus the
    first page's body text, and classify by keyword. PDF metadata Title is
    frequently export junk (e.g. an InDesign filename), so the body text is
    checked too rather than relying on the title alone.
    Falls back to the raw extracted title (or 'Other') so nothing is silently dropped.
    """
    filename = pdf_path.name
    if _BROCHURE_RE.search(filename):
        return "Sales Brochure"

    title, first_page_text = _extract_pdf_title_and_text(pdf_path)
    haystack = f"{filename}\n{title}\n{first_page_text[:1000]}"

    if _BROCHURE_RE.search(haystack):
        return "Sales Brochure"
    if _CIS_RE.search(haystack):
        return "Customer Information Sheet"
    if _POLICY_DOC_RE.search(haystack):
        return "Policy Document"

    return title if title else "Other"


def iter_product_pdfs(product_pdfs_root: Path) -> Iterator[Tuple[Path, Dict]]:
    """
    Walk PRODUCT_PDFS_ROOT for the folders listed in PRODUCT_TYPE_FOLDERS and
    yield (pdf_path, metadata) pairs. Each folder's PDFs sit directly inside it
    (no sub-folders) — metadata is just product_type + an auto-detected doc_type.
    """
    for folder_name, product_type in PRODUCT_TYPE_FOLDERS.items():
        product_dir = product_pdfs_root / folder_name
        if not product_dir.is_dir():
            print(f"Skipping {folder_name} — folder not found under {product_pdfs_root}")
            continue

        pdf_paths = sorted(product_dir.glob("*.pdf"))
        print(f"Opened folder: {folder_name}  (product_type={product_type}, {len(pdf_paths)} PDF(s))")

        for pdf_path in pdf_paths:
            yield pdf_path, {
                "product_type": product_type,
                "doc_type": identify_doc_type(pdf_path),
            }
