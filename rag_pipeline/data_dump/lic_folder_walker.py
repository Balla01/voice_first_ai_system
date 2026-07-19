"""
Real LIC folder-tree walker (constants.COMPLETE_FOLDER_STRUCTURE=True).

Layout (see constants.py for the exact paths):
  LIC_DATA_ROOT/insurance-plans/<category>/<plan-folder>/*.pdf
  LIC_DATA_ROOT/pension-plans/<plan-folder>/*.pdf          (no category level)

A "plan folder" is any directory that directly contains one or more PDFs —
detected structurally (via os.walk) rather than by assuming a fixed depth, so
insurance-plans' extra category level and pension-plans' flatter layout are
both handled by the same code.

<plan-folder> is named like "lic-amritbaal-774-512n365v02": the trailing
hyphen-segments are the LIC plan number ("774") and UIN ("512n365v02"). Rider
folders (e.g. "lic-accident-benefit-rider-512b203v03") have a UIN but no plan
number, since riders aren't sold under their own plan number — plan_no is
None in that case, not a guess.
"""

import os
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from constants import LIC_DATA_ROOT, LIC_PRODUCT_TYPE_FOLDERS
from data_dump.lic_metadata import identify_doc_type

# e.g. "512n365v02", "512N339V03", "512L347V01" — 3 digits, 1 letter, 3 digits,
# 'v' + 2-3 digits. Matches every UIN observed in the actual LIC_DATA_ROOT tree.
_UIN_RE = re.compile(r'^\d{3}[A-Za-z]\d{3}[Vv]\d{2,3}$')
_PLAN_NO_RE = re.compile(r'^\d{2,4}$')


def parse_plan_folder_name(folder_name: str) -> Dict[str, Optional[str]]:
    """
    "lic-amritbaal-774-512n365v02" -> {"plan_no": "774", "uin_no": "512n365v02"}
    "lic-accident-benefit-rider-512b203v03" -> {"plan_no": None, "uin_no": "512b203v03"}
    (no UIN-shaped trailing segment) -> {"plan_no": None, "uin_no": None}
    """
    parts = folder_name.split("-")
    uin_no = parts[-1] if parts and _UIN_RE.match(parts[-1]) else None
    plan_no = None
    if uin_no and len(parts) >= 2 and _PLAN_NO_RE.match(parts[-2]):
        plan_no = parts[-2]
    return {"plan_no": plan_no, "uin_no": uin_no}


def iter_complete_structure_pdfs(lic_data_root: Path = LIC_DATA_ROOT) -> Iterator[Tuple[Path, Dict]]:
    """
    Walk LIC_PRODUCT_TYPE_FOLDERS under lic_data_root and yield (pdf_path, metadata)
    pairs, metadata = {product_type, doc_type, category, plan_folder, plan_no, uin_no}.
    category is None for product roots with no category level (e.g. pension-plans).
    """
    for folder_name, product_type in LIC_PRODUCT_TYPE_FOLDERS.items():
        product_root = lic_data_root / folder_name
        if not product_root.is_dir():
            print(f"Skipping {folder_name} — folder not found under {lic_data_root}")
            continue
        print(f"Scanning product folder: {folder_name}  (product_type={product_type})")

        for dirpath, _dirnames, filenames in os.walk(product_root):
            pdf_names = sorted(f for f in filenames if f.lower().endswith(".pdf"))
            if not pdf_names:
                continue

            plan_dir = Path(dirpath)
            rel_parts = plan_dir.relative_to(product_root).parts
            category = rel_parts[0] if len(rel_parts) > 1 else None
            plan_folder = rel_parts[-1]
            plan_ids = parse_plan_folder_name(plan_folder)

            print(f"  Opened folder: {folder_name}/{plan_dir.relative_to(product_root).as_posix()}  ({len(pdf_names)} PDF(s))")

            for pdf_name in pdf_names:
                pdf_path = plan_dir / pdf_name
                yield pdf_path, {
                    "product_type": product_type,
                    "doc_type":     identify_doc_type(pdf_path),
                    "category":     category,
                    "plan_folder":  plan_folder,
                    "plan_no":      plan_ids["plan_no"],
                    "uin_no":       plan_ids["uin_no"],
                }
