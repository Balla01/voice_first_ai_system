"""
LIC folder-tree walker + metadata extraction.

Layout handled:
  LIC_ROOT/insurance-plans/<category>/<sub_category-plan_no-UIN>/*.pdf
  LIC_ROOT/pension-plans/<sub_category-plan_no-UIN>/*.pdf   (no category level)

For each PDF this yields the folder-derived metadata (product_type, category,
sub_category, plan_no, uin) plus a doc_type identified from the filename/title.
"""

import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pdfplumber

from constants import LIC_PRODUCT_TYPES, LIC_PRODUCT_TYPES_WITH_CATEGORY

# UIN: 3 digits, 1 letter, 3 digits, 1 letter, 2 digits (e.g. 512N337V07)
_UIN_RE = re.compile(r'-(\d{3}[A-Za-z]\d{3}[A-Za-z]\d{2})$')
_PLAN_NO_RE = re.compile(r'-(\d{3})$')

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


def _humanize(slug: str) -> str:
    """'money_back_plans' / 'new-endowment-plan' -> 'Money Back Plans' / 'New Endowment Plan'"""
    words = [w for w in re.split(r'[-_]+', slug) if w]
    return " ".join(w.capitalize() for w in words)


def parse_subfolder_name(folder_name: str) -> Tuple[str, str, str]:
    """
    Extract (sub_category, plan_no, uin) from a LIC sub-category folder name.

    'lic-nivesh-plus-749-512L317V02'          -> ('Nivesh Plus', '749', '512L317V02')
    'lic-accident-benefit-rider-512b203v03'   -> ('Accident Benefit Rider', '', '512B203V03')
    plan_no is '' when the folder has no plan number (e.g. rider plans).
    """
    name = folder_name.strip()

    uin = ""
    uin_match = _UIN_RE.search(name)
    if uin_match:
        uin = uin_match.group(1).upper()
        name = name[:uin_match.start()]

    plan_no = ""
    plan_match = _PLAN_NO_RE.search(name)
    if plan_match:
        plan_no = plan_match.group(1)
        name = name[:plan_match.start()]

        # Drop the '-plan-no' / '-plan' label that just announces the plan
        # number in the folder-naming convention (e.g. 'jeevan-akshay-vii-plan-no'
        # -> 'jeevan-akshay-vii'; 'jeevan-labh-plan' -> 'jeevan-labh').
        name = re.sub(r'-plan-no$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'-plan$', '', name, flags=re.IGNORECASE)

    if name.lower().startswith("lic-"):
        name = name[4:]
    name = name.strip("-")

    return _humanize(name), plan_no, uin


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


def _pdfs_in_subcategory(sub_dir: Path, product_type: str, category: str) -> Iterator[Tuple[Path, Dict]]:
    sub_category, plan_no, uin = parse_subfolder_name(sub_dir.name)

    for pdf_path in sorted(sub_dir.glob("*.pdf")):
        yield pdf_path, {
            "product_type": product_type,
            "category": category,
            "sub_category": sub_category,
            "plan_no": plan_no,
            "uin": uin,
            "doc_type": identify_doc_type(pdf_path),
            "source_folder": sub_dir.name,
        }


def iter_lic_pdfs(lic_root: Path) -> Iterator[Tuple[Path, Dict]]:
    """
    Walk LIC_ROOT for the supported product types (LIC_PRODUCT_TYPES) and yield
    (pdf_path, metadata) pairs. Other sibling product-type folders are skipped.
    """
    for product_type in LIC_PRODUCT_TYPES:
        product_dir = lic_root / product_type
        if not product_dir.is_dir():
            continue

        if product_type in LIC_PRODUCT_TYPES_WITH_CATEGORY:
            for category_dir in sorted(d for d in product_dir.iterdir() if d.is_dir()):
                category = _humanize(category_dir.name)
                for sub_dir in sorted(d for d in category_dir.iterdir() if d.is_dir()):
                    yield from _pdfs_in_subcategory(sub_dir, product_type, category)
        else:
            for sub_dir in sorted(d for d in product_dir.iterdir() if d.is_dir()):
                yield from _pdfs_in_subcategory(sub_dir, product_type, category="")
