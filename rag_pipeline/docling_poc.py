"""
PDF content segregation using Docling (docling-project/docling).

Docling runs each page through its layout model + the TableFormer structure
model, so every element (paragraph, heading, table, ...) comes back as a
separate structured item tagged with its own page number. Tables are recognized
as real row/column grids by TableFormer, not scraped out of the text stream.

This script produces ONE json (`content.json`) that holds, for the whole PDF:

  1. Page-wise content, individually  -> `pages[]`
       each page's paragraph text and the tables that appear on it.

  2. Normal text vs table text, segregated
       a page's `paragraphs` / `text` never contain table cell text, and the
       tables are kept as structured rows — the two never contaminate each other.

  3. Multi-page tables stitched into one logical table  -> `logical_tables[]`
       a table whose region continues onto page 2, 3, 4, ... is merged into a
       single logical table, with the first page's header carried forward to the
       headerless continuation pages.

  4. Separate table regions kept individual
       tables on pages 1-4 and a different table on pages 5-7 come out as two
       distinct entries in `logical_tables[]`, each with its own page span.
       A new logical table starts when a fresh header appears or the column
       count changes.

Note: stitching relies on TableFormer detecting a consistent column count across
the continuation pages. On documents where its per-page column detection drifts
(long, dense, borderless tables), continuations may not merge cleanly — that is
a model limitation, not a bug in the stitching logic.

Install:
    pip install docling

Usage:
    python docling_table_extractor.py <path_to_pdf>
    python docling_table_extractor.py <path_to_pdf> --output-dir out/docling --table-mode accurate
    python docling_table_extractor.py <path_to_pdf> --pages 1-10 --ocr
    python docling_table_extractor.py <path_to_pdf> --no-stitch     # keep tables per-page, no merging
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem


# ── Converter ─────────────────────────────────────────────────────────────────

def build_converter(table_mode: str, do_ocr: bool) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
    )
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.do_ocr = do_ocr

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def parse_page_range(pages: Optional[str]) -> Optional[Tuple[int, int]]:
    if not pages:
        return None
    start, _, end = pages.partition("-")
    return (int(start), int(end or start))


def page_number(item) -> Optional[int]:
    return item.prov[0].page_no if item.prov else None


# ── Cell / table normalization ────────────────────────────────────────────────

def _cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _normalize_header(df: pd.DataFrame) -> List[str]:
    return [_cell(c) for c in df.columns]


def _normalize_rows(df: pd.DataFrame) -> List[List[str]]:
    return [[_cell(v) for v in row] for row in df.itertuples(index=False, name=None)]


def _is_positional_header(header: List[str]) -> bool:
    """
    True when Docling could NOT find a real header row and fell back to
    positional names ('0', '1', '2', ...). Such a table is a continuation
    candidate — its real content is entirely in the data rows.
    """
    return all(h == str(i) for i, h in enumerate(header))


def rows_to_markdown(header: Optional[List[str]], rows: List[List[str]]) -> str:
    """Plain pipe-table renderer (no `tabulate` dep). Falls back to positional header."""
    ncols = len(header) if header else (max((len(r) for r in rows), default=0))
    head = header if header else [str(i) for i in range(ncols)]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in rows:
        padded = list(row) + [""] * (ncols - len(row))
        lines.append("| " + " | ".join(padded[:ncols]) + " |")
    return "\n".join(lines)


# ── Structured table types ────────────────────────────────────────────────────

@dataclass
class PageTable:
    """One table exactly as Docling found it on a single page."""
    page: int
    page_table_index: int
    header: List[str]
    rows: List[List[str]]

    @property
    def num_cols(self) -> int:
        return len(self.header)


@dataclass
class LogicalTable:
    """One table region, possibly stitched across several pages."""
    table_id: int
    num_cols: int
    header: Optional[List[str]] = None
    rows: List[List[str]] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)

    @property
    def page_span(self) -> str:
        if not self.pages:
            return "?"
        lo, hi = min(self.pages), max(self.pages)
        return f"{lo}" if lo == hi else f"{lo}-{hi}"

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "pages": sorted(set(self.pages)),
            "page_span": self.page_span,
            "spans_multiple_pages": len(set(self.pages)) > 1,
            "num_cols": self.num_cols,
            "num_rows": len(self.rows),
            "header": self.header,
            "rows": self.rows,
            "markdown": rows_to_markdown(self.header, self.rows),
        }


# ── Stitching ─────────────────────────────────────────────────────────────────

def stitch_tables(page_tables: List[PageTable]) -> List[LogicalTable]:
    """
    Merge consecutive page-tables into logical tables, carrying the header
    forward onto headerless continuations.

    A new logical table starts when:
      - nothing is open yet, or
      - the table has a REAL (non-positional) header  -> a new table begins, or
      - the column count differs from the open table  -> a different table.
    Otherwise the table is a continuation: its data rows are appended and the
    open table's header applies.
    """
    logical: List[LogicalTable] = []
    current: Optional[LogicalTable] = None

    for pt in page_tables:
        positional = _is_positional_header(pt.header)
        start_new = (
            current is None
            or not positional
            or pt.num_cols != current.num_cols
        )

        if start_new:
            if current is not None:
                logical.append(current)
            if positional:
                # New table with no real header (rare — e.g. the very first
                # table already lacks one). All content is data.
                current = LogicalTable(
                    table_id=len(logical), num_cols=pt.num_cols,
                    header=None, rows=list(pt.rows), pages=[pt.page],
                )
            else:
                current = LogicalTable(
                    table_id=len(logical), num_cols=pt.num_cols,
                    header=list(pt.header), rows=list(pt.rows), pages=[pt.page],
                )
        else:
            current.rows.extend(pt.rows)
            current.pages.append(pt.page)

    if current is not None:
        logical.append(current)

    return logical


# ── Extraction ────────────────────────────────────────────────────────────────

def extract(pdf_path: Path, output_dir: Path, table_mode: str, do_ocr: bool,
            pages: Optional[str], stitch: bool) -> dict:
    page_range = parse_page_range(pages)
    converter = build_converter(table_mode, do_ocr)

    print(f"Converting {pdf_path.name} with Docling (table-mode={table_mode}, ocr={do_ocr}) ...")
    convert_kwargs = {"page_range": page_range} if page_range else {}
    result = converter.convert(str(pdf_path), **convert_kwargs)
    doc = result.document

    # Walk items in reading order; bucket paragraphs vs tables per page.
    # page_tables keeps global document order (needed for correct stitching).
    pages_content = {}          # page_no -> {"paragraphs": [str]}
    page_table_counter = {}     # page_no -> running table index on that page
    page_tables: List[PageTable] = []

    for item, _level in doc.iterate_items():
        pno = page_number(item)
        if pno is None:
            continue
        bucket = pages_content.setdefault(pno, {"paragraphs": []})

        if isinstance(item, TableItem):
            df = item.export_to_dataframe(doc=doc)
            idx = page_table_counter.get(pno, 0) + 1
            page_table_counter[pno] = idx
            page_tables.append(PageTable(
                page=pno, page_table_index=idx,
                header=_normalize_header(df), rows=_normalize_rows(df),
            ))
        elif hasattr(item, "text") and item.text and item.text.strip():
            bucket["paragraphs"].append(item.text.strip())

    # Group the per-page tables so we can list them under each page too.
    tables_by_page = {}
    for pt in page_tables:
        tables_by_page.setdefault(pt.page, []).append(pt)

    # ── Requirement 4: page-wise content, individually ──
    pages_out = []
    for pno in sorted(pages_content):
        paragraphs = pages_content[pno]["paragraphs"]
        pages_out.append({
            "page": pno,
            "text": "\n\n".join(paragraphs),          # normal text only
            "paragraphs": paragraphs,
            "tables_on_page": [
                {
                    "page_table_index": pt.page_table_index,
                    "num_cols": pt.num_cols,
                    "num_rows": len(pt.rows),
                    "header": pt.header,
                    "rows": pt.rows,
                    "markdown": rows_to_markdown(pt.header, pt.rows),
                }
                for pt in tables_by_page.get(pno, [])
            ],
        })

    # ── Requirements 1 + 3: stitched multi-page tables, kept individual ──
    if stitch:
        logical_tables = stitch_tables(page_tables)
    else:
        logical_tables = [
            LogicalTable(table_id=i, num_cols=pt.num_cols,
                         header=pt.header, rows=pt.rows, pages=[pt.page])
            for i, pt in enumerate(page_tables)
        ]

    content = {
        "source_file": pdf_path.name,
        "table_mode": table_mode,
        "ocr": do_ocr,
        "stitched": stitch,
        "total_pages": len(pages_out),
        "total_page_tables": len(page_tables),
        "total_logical_tables": len(logical_tables),
        "pages": pages_out,
        "logical_tables": [lt.to_dict() for lt in logical_tables],
    }

    _write_outputs(content, logical_tables, pages_out, output_dir)
    return content


def _write_outputs(content: dict, logical_tables: List[LogicalTable],
                   pages_out: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary artifact: everything in one JSON.
    content_path = output_dir / "content.json"
    content_path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")

    # Convenience side files: per-page text, per-logical-table CSV.
    text_dir = output_dir / "text"
    table_dir = output_dir / "tables"
    text_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    for page in pages_out:
        (text_dir / f"page_{page['page']:03d}.txt").write_text(page["text"], encoding="utf-8")

    for lt in logical_tables:
        d = lt.to_dict()
        header = d["header"] or [str(i) for i in range(d["num_cols"])]
        csv_lines = [",".join(_csv_escape(c) for c in header)]
        csv_lines += [",".join(_csv_escape(c) for c in (row + [""] * (d["num_cols"] - len(row)))) for row in d["rows"]]
        (table_dir / f"table{lt.table_id + 1}_pages{lt.page_span}.csv").write_text(
            "\n".join(csv_lines), encoding="utf-8"
        )

    print(f"\nPages processed      : {content['total_pages']}")
    print(f"Raw per-page tables  : {content['total_page_tables']}")
    print(f"Logical tables       : {content['total_logical_tables']}")
    for lt in logical_tables:
        span = lt.page_span
        multi = "  (stitched across pages)" if len(set(lt.pages)) > 1 else ""
        hdr = "with header" if lt.header else "no header"
        print(f"   - table {lt.table_id + 1}: pages {span} | {len(lt.rows)} rows x {lt.num_cols} cols | {hdr}{multi}")
    print(f"\nJSON (all content)   : {content_path.resolve()}")
    print(f"Per-page text        : {text_dir.resolve()}")
    print(f"Per-table CSV        : {table_dir.resolve()}")


def _csv_escape(cell: str) -> str:
    cell = cell.replace("\n", " ")
    if any(ch in cell for ch in [",", '"']):
        return '"' + cell.replace('"', '""') + '"'
    return cell


def main():
    parser = argparse.ArgumentParser(description="Segregate PDF content (text vs tables) with Docling, stitching multi-page tables")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default="out/docling", help="Where to write content.json + side files")
    parser.add_argument(
        "--table-mode", choices=["fast", "accurate"], default="accurate",
        help="TableFormer mode - accurate is slower but recovers merged headers/spans better",
    )
    parser.add_argument("--ocr", action="store_true", help="Enable OCR (needed for scanned/image-only PDFs)")
    parser.add_argument("--pages", default=None, help="Page range to process, e.g. 1-10 (1-indexed, inclusive)")
    parser.add_argument("--no-stitch", action="store_true", help="Keep tables per-page; do not merge multi-page tables")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    extract(pdf_path, Path(args.output_dir), args.table_mode, args.ocr, args.pages, stitch=not args.no_stitch)


if __name__ == "__main__":
    main()
 