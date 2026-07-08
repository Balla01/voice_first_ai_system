"""
Standalone table extractor using pdfplumber.

Usage:
    python extract_tables.py <path_to_pdf>
    python extract_tables.py <path_to_pdf> --output-dir out --format csv
    python extract_tables.py <path_to_pdf> --output-dir out --format json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pdfplumber

Table = List[List[Optional[str]]]


def extract_tables(pdf_path: str) -> Iterator[Tuple[int, int, Table]]:
    """Yields (page_number, table_index, table) for every table found in the PDF (1-indexed)."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table_idx, table in enumerate(page.extract_tables(), start=1):
                yield page_num, table_idx, table


def save_table_csv(table: Table, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in table:
            writer.writerow("" if cell is None else cell for cell in row)


def print_table(table: Table) -> None:
    for row in table:
        print(["" if cell is None else cell for cell in row])


def main():
    parser = argparse.ArgumentParser(description="Extract tables from a PDF using pdfplumber")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default=None, help="If set, saves extracted tables here")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    json_tables = []
    total = 0

    for page_num, table_idx, table in extract_tables(str(pdf_path)):
        total += 1
        print(f"\n=== Page {page_num}, Table {table_idx} ({len(table)} rows) ===")
        print_table(table)

        if output_dir:
            if args.format == "csv":
                save_table_csv(table, output_dir / f"page{page_num}_table{table_idx}.csv")
            else:
                json_tables.append({"page": page_num, "table_index": table_idx, "rows": table})

    if output_dir and args.format == "json":
        with open(output_dir / "tables.json", "w", encoding="utf-8") as f:
            json.dump(json_tables, f, indent=2, ensure_ascii=False)

    print(f"\nTotal tables found: {total}")
    if output_dir:
        print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
