"""
Standalone table extractor using pdfplumber, with multi-page table stitching.

Core problem this solves:
    A single logical table often spans several pages. The header row appears
    only once (page 1); pages 2, 3, ... continue the same table with data rows
    and NO header of their own. Extracting page-by-page would leave those
    continuation pages headerless (columns become 0/1/2/... with no meaning).

What this does:
    Consecutive page-tables are merged into one "logical table" and the page-1
    header is carried forward to the headerless continuations. A new logical
    table is started when either:
      - the first row of a page-table looks like a header (a new table begins), or
      - the column count changes (structurally a different table).

    So: header on page 1, plain data on pages 2 & 3  -> ONE table (header reused).
        A fresh header (or different column count) on page 4 -> a NEW table.

    This header/data detection is a heuristic (pdfplumber gives rows, not a
    "this is a heading" flag). Pass --no-merge to disable stitching and get the
    old raw per-page behavior.

Usage:
    python extract_tables.py <path_to_pdf>
    python extract_tables.py <path_to_pdf> --output-dir out --format csv
    python extract_tables.py <path_to_pdf> --output-dir out --format json
    python extract_tables.py <path_to_pdf> --no-merge          # raw per-page, no stitching
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pdfplumber

Row = List[Optional[str]]
Table = List[Row]

# A cell that is essentially a number (incl. currency/percent/thousands separators).
_NUMERIC_RE = re.compile(r"^[-+]?[₹$]?\s*[\d][\d,.\s]*%?$")

# When True, also write the human-inspectable side artifacts (per-table CSVs /
# tables.json, per-page prose text/ + all_text.txt). When False, chunks.json is
# the ONLY output — that's the production/return artifact. Overridable per-run
# with --debug (see main()).
DEBUG = False

# ── Chunking defaults (see the --chunk-* CLI flags) ───────────────────────────
DEFAULT_CHUNK_TOKENS = 400     # common token budget shared by prose + tables (rule: ~300-500)
DEFAULT_CHUNK_OVERLAP = 50     # overlap carried between chunks when the boundary falls in prose
DEFAULT_TABLE_MAX_ROWS = 0     # optional hard row cap per table piece (0 = token-driven only)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ── Low-level per-page extraction ─────────────────────────────────────────────

def extract_page_tables(pdf_path: str) -> Iterator[Tuple[int, int, Table]]:
    """Yields (page_number, table_index_on_page, table) for every table (1-indexed)."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table_idx, table in enumerate(page.extract_tables(), start=1):
                if table:
                    yield page_num, table_idx, table


def extract_page_prose(pdf_path: str) -> Iterator[Tuple[int, str]]:
    """
    Yields (page_number, prose_text) for every page, where prose_text is the
    page's normal text with the TABLE REGIONS REMOVED.

    How: pdfplumber locates each table's bounding box (page.find_tables()),
    then text is extracted only from the objects whose center falls OUTSIDE
    every table box — so table cell text never leaks into the prose, and the
    prose is exactly the "normal context" around the tables.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            bboxes = [t.bbox for t in page.find_tables()]

            if not bboxes:
                text = page.extract_text() or ""
            else:
                def _outside(obj, _bboxes=bboxes):
                    cx = (obj["x0"] + obj["x1"]) / 2
                    cy = (obj["top"] + obj["bottom"]) / 2
                    return not any(
                        x0 <= cx <= x1 and top <= cy <= bottom
                        for (x0, top, x1, bottom) in _bboxes
                    )
                text = page.filter(_outside).extract_text() or ""

            yield page_num, text.strip()


# ── Header / continuation heuristics ──────────────────────────────────────────

def _clean(cell: Optional[str]) -> str:
    return "" if cell is None else str(cell).strip()


def _num_cols(table: Table) -> int:
    return max((len(r) for r in table), default=0)


def _is_numeric(cell: str) -> bool:
    return bool(_NUMERIC_RE.match(cell.replace("\n", " ").strip()))


def looks_like_header(row: Row, num_cols: int) -> bool:
    """
    Heuristic: a header row fills (nearly) all of its columns with non-numeric
    text. Continuation / data rows in these documents typically leave the
    leading columns blank (e.g. the "Sl. no." and "Title" cells are empty
    because the row just continues the previous item's description), so a
    partially-filled row is treated as data, not a header.

    A row is treated as a header when:
      - at least 75% of its columns are non-empty, AND
      - at most 40% of the non-empty cells are numeric.
    """
    if num_cols == 0:
        return False
    cells = [_clean(c) for c in row]
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    if len(non_empty) / num_cols < 0.75:
        return False
    numeric = sum(1 for c in non_empty if _is_numeric(c))
    return numeric <= 0.4 * len(non_empty)


# ── Logical (multi-page) table ────────────────────────────────────────────────

@dataclass
class LogicalTable:
    index: int
    num_cols: int
    header: Optional[Row] = None
    rows: Table = field(default_factory=list)   # data rows only (header excluded)
    pages: List[int] = field(default_factory=list)

    @property
    def all_rows(self) -> Table:
        return ([self.header] if self.header is not None else []) + self.rows

    @property
    def page_span(self) -> str:
        if not self.pages:
            return "?"
        lo, hi = min(self.pages), max(self.pages)
        return f"{lo}" if lo == hi else f"{lo}-{hi}"


def group_tables(pdf_path: str) -> List[LogicalTable]:
    """
    Walk page-tables in reading order and stitch continuations onto the
    preceding logical table, carrying its header forward.
    """
    logical: List[LogicalTable] = []
    current: Optional[LogicalTable] = None

    for page_num, _table_idx, table in extract_page_tables(pdf_path):
        ncols = _num_cols(table)
        first_row = table[0]
        header_like = looks_like_header(first_row, ncols)

        # Start a new logical table when there's nothing open yet, when a fresh
        # header appears, or when the column count doesn't match the open one.
        start_new = (
            current is None
            or header_like
            or ncols != current.num_cols
        )

        if start_new:
            if current is not None:
                logical.append(current)
            if header_like:
                current = LogicalTable(
                    index=len(logical), num_cols=ncols,
                    header=first_row, rows=table[1:], pages=[page_num],
                )
            else:
                # New table but no detectable header (e.g. first table in the
                # doc already lacks one) — keep all rows as data.
                current = LogicalTable(
                    index=len(logical), num_cols=ncols,
                    header=None, rows=list(table), pages=[page_num],
                )
        else:
            # Continuation: same column count, no header of its own -> the open
            # table's header applies. Every row here is data.
            current.rows.extend(table)
            if page_num not in current.pages:
                current.pages.append(page_num)

    if current is not None:
        logical.append(current)

    return logical


# ── Output helpers ────────────────────────────────────────────────────────────

def save_table_csv(rows: Table, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(_clean(cell) for cell in row)


def print_table(rows: Table) -> None:
    for row in rows:
        print([_clean(cell) for cell in row])


# ── Token counting ────────────────────────────────────────────────────────────

_HF_TOKENIZER = None
_USE_EXACT_TOKENS = False


def init_tokenizer(exact: bool) -> None:
    """
    Choose how tokens are counted for the token caps.
      exact=False (default): fast heuristic (~0.75 words per token). No deps.
      exact=True: load the BAAI/bge-m3 tokenizer so counts match the embedding
                  model used by the main pipeline. Slower to start; needs
                  transformers + the model cached locally.
    """
    global _HF_TOKENIZER, _USE_EXACT_TOKENS
    _USE_EXACT_TOKENS = exact
    if exact:
        from transformers import AutoTokenizer
        _HF_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")


def count_tokens(text: str) -> int:
    if _USE_EXACT_TOKENS and _HF_TOKENIZER is not None:
        return len(_HF_TOKENIZER.encode(text, add_special_tokens=False))
    words = len(text.split())
    return max(1, round(words / 0.75)) if words else 0


def _page_span(pages: List[int]) -> str:
    if not pages:
        return "?"
    lo, hi = min(pages), max(pages)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


# ── Prose chunking (rule: logical section, ~300-500 tokens, ~50 overlap) ──────

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _hard_split_words(sentence: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Fallback for a single sentence longer than max_tokens: split on words."""
    words = sentence.split()
    pieces, start = [], 0
    # approx words per chunk from the token budget (heuristic inverse)
    step = max(int(max_tokens * 0.75), 1)
    ov = max(int(overlap_tokens * 0.75), 0)
    while start < len(words):
        piece = " ".join(words[start : start + step])
        pieces.append(piece)
        start += max(step - ov, 1)
    return pieces


# ── Table markdown + chunking ─────────────────────────────────────────────────

def _md_cell(cell: Optional[str]) -> str:
    return _clean(cell).replace("\n", " ").replace("|", "\\|")


def _md_header_block(header: Optional[Row], ncols: int) -> str:
    """The two markdown header lines (labels + separator), repeated in every table chunk."""
    head = header if header else [str(i) for i in range(ncols)]
    return (
        "| " + " | ".join(_md_cell(c) for c in head) + " |\n"
        "| " + " | ".join(["---"] * ncols) + " |"
    )


def _md_row_line(row: Row, ncols: int) -> str:
    padded = list(row) + [""] * (ncols - len(row))
    return "| " + " | ".join(_md_cell(c) for c in padded[:ncols]) + " |"


def render_markdown(header: Optional[Row], rows: Table) -> str:
    """Render header + rows as a GitHub-flavored pipe table."""
    ncols = len(header) if header else max((len(r) for r in rows), default=0)
    lines = [_md_header_block(header, ncols)]
    lines += [_md_row_line(row, ncols) for row in rows]
    return "\n".join(lines)


# ── Build + write chunks.json ─────────────────────────────────────────────────

def _is_real_table(rows: Table) -> bool:
    """
    True only for a genuine data grid. Rejects pdfplumber's false positives on
    visually-designed pages (brochures), where decorative rectangles/boxes get
    read as tables: those come back as rows with text dumped into a SINGLE
    column (the other cells empty), whereas a real table — including a
    continuation page of a multi-page table — has rows spanning >= 2 columns.

    A rejected "table" is not treated as a table at all; its text falls back
    into the surrounding prose flow (see extract_page_blocks).
    """
    if not rows:
        return False
    ncols = max((len(r) for r in rows), default=0)
    if ncols < 2:
        return False

    def filled(r: Row) -> int:
        return sum(1 for c in r if c is not None and str(c).strip())

    multi_col_rows = sum(1 for r in rows if filled(r) >= 2)
    # Keep only if at least half the rows (and >= 1) are genuinely multi-column.
    return multi_col_rows >= max(1, (len(rows) + 1) // 2)


def extract_page_blocks(page, page_num: int) -> List[dict]:
    """
    Return this page's content as an ordered list of blocks, top-to-bottom, so
    text and tables keep their real reading order. Each block is either:
      {"kind": "prose", "page", "top", "text"}
      {"kind": "table", "page", "top", "rows"}

    Prose is split at table boundaries: text above the first table, text between
    tables, and text below the last table each become separate prose blocks —
    so a page laid out as text/table/text/table yields four ordered blocks.

    Only tables that pass _is_real_table() are treated as tables; pdfplumber
    false positives (decorative boxes) are ignored here so their text flows
    into the prose blocks instead of becoming junk table markdown.
    """
    kept = []
    for t in page.find_tables():
        rows = t.extract()
        if _is_real_table(rows):
            kept.append((t, rows))
    kept.sort(key=lambda tr: tr[0].bbox[1])   # by top edge
    bboxes = [t.bbox for t, _ in kept]

    def region_prose(y_lo: float, y_hi: float) -> str:
        def keep(obj):
            cy = (obj["top"] + obj["bottom"]) / 2
            if not (y_lo <= cy < y_hi):
                return False
            cx = (obj["x0"] + obj["x1"]) / 2
            return not any(x0 <= cx <= x1 and top <= cy <= bottom for (x0, top, x1, bottom) in bboxes)
        return (page.filter(keep).extract_text() or "").strip()

    blocks: List[dict] = []

    if not kept:
        txt = (page.extract_text() or "").strip()
        if txt:
            blocks.append({"kind": "prose", "page": page_num, "top": 0.0, "text": txt})
        return blocks

    above = region_prose(0.0, kept[0][0].bbox[1])
    if above:
        blocks.append({"kind": "prose", "page": page_num, "top": 0.0, "text": above})

    for i, (t, rows) in enumerate(kept):
        blocks.append({"kind": "table", "page": page_num, "top": t.bbox[1], "rows": rows})
        y_hi = kept[i + 1][0].bbox[1] if i + 1 < len(kept) else float(page.height)
        between = region_prose(t.bbox[3], y_hi)
        if between:
            blocks.append({"kind": "prose", "page": page_num, "top": t.bbox[3], "text": between})

    return blocks


def _ordered_elements(pdf_path: Path) -> List[dict]:
    """
    Document content in reading order, with multi-page tables already stitched.
    Each element is:
      {"kind": "prose", "page", "text"}
      {"kind": "table", "table_id", "pages", "header", "rows"}
    """
    all_blocks: List[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            all_blocks.extend(extract_page_blocks(page, pno))

    elements: List[dict] = []
    open_tbl: Optional[LogicalTable] = None
    table_count = 0

    def finalize():
        nonlocal open_tbl
        if open_tbl is not None:
            elements.append({
                "kind": "table",
                "table_id": open_tbl.index + 1,
                "pages": sorted(set(open_tbl.pages)),
                "header": open_tbl.header,
                "rows": open_tbl.rows,
            })
            open_tbl = None

    for blk in all_blocks:
        if blk["kind"] == "prose":
            finalize()
            elements.append({"kind": "prose", "page": blk["page"], "text": blk["text"]})
            continue

        rows = blk["rows"]
        ncols = _num_cols(rows) if rows else 0
        header_like = bool(rows) and looks_like_header(rows[0], ncols)

        if open_tbl is not None and ncols == open_tbl.num_cols and not header_like:
            open_tbl.rows.extend(rows)
            open_tbl.pages.append(blk["page"])
        else:
            finalize()
            table_count += 1
            if header_like:
                header, data_rows = rows[0], rows[1:]
            else:
                header, data_rows = None, rows
            open_tbl = LogicalTable(index=table_count - 1, num_cols=ncols,
                                    header=header, rows=list(data_rows), pages=[blk["page"]])

    finalize()
    return elements


def build_chunks(pdf_path: Path, chunk_tokens: int, overlap_tokens: int, table_max_rows: int) -> List[dict]:
    """
    Pack the reading-order element stream into chunks under ONE common token
    budget (chunk_tokens). Prose and tables share chunks:

      - text and the start of a following table can sit in the same chunk;
      - when a table doesn't fully fit, it fills the chunk, then CONTINUES in
        the next chunk with its header repeated ("that header logic");
      - text after the table's remainder joins that next chunk (budget allowing).

    A table row is never split. Prose carries ~overlap_tokens between chunks
    when a boundary falls in prose (not when it falls mid-table — the repeated
    header is the continuation cue there instead).
    """
    elements = _ordered_elements(pdf_path)

    records: List[dict] = []
    parts: List[dict] = []     # current chunk's ordered parts
    tok = 0
    pages: set = set()

    def render_parts(ps: List[dict]) -> str:
        segs = []
        for p in ps:
            if p["type"] == "prose":
                segs.append(" ".join(s for _, s in p["sents"]))
            else:
                segs.append(render_markdown(p["header"], p["rows"]))
        return "\n\n".join(segs)

    def summarize(ps: List[dict]) -> List[dict]:
        out = []
        for p in ps:
            if p["type"] == "prose":
                out.append({
                    "type": "prose",
                    "pages": sorted({pg for pg, _ in p["sents"]}),
                    "text": " ".join(s for _, s in p["sents"]),
                })
            else:
                out.append({
                    "type": "table",
                    "table_id": p["table_id"],
                    "pages": p["pages"],
                    "num_rows": len(p["rows"]),
                    "header": p["header"],
                    "markdown": render_markdown(p["header"], p["rows"]),
                })
        return out

    def flush(overlap: bool):
        nonlocal parts, tok, pages
        if not parts:
            return
        content = render_parts(parts)
        types = {p["type"] for p in parts}
        records.append({
            "type": "mixed" if len(types) > 1 else next(iter(types)),
            "pages": sorted(pages),
            "page_span": _page_span(sorted(pages)),
            "token_count": count_tokens(content),
            "parts": summarize(parts),
            "content": content,
        })
        carry, cpages = [], set()
        if overlap and parts[-1]["type"] == "prose":
            keep, ct = [], 0
            for u in reversed(parts[-1]["sents"]):
                t = count_tokens(u[1])
                if ct + t > overlap_tokens:
                    break
                keep.insert(0, u)
                ct += t
            if keep:
                carry = [{"type": "prose", "sents": keep}]
                cpages = {pg for pg, _ in keep}
        parts = carry
        tok = sum(count_tokens(s) for p in parts for _, s in p["sents"]) if parts else 0
        pages = cpages

    def add_prose(page: int, text: str):
        nonlocal tok, pages
        sents: List[str] = []
        for s in _split_sentences(text):
            if count_tokens(s) > chunk_tokens:
                sents.extend(_hard_split_words(s, chunk_tokens, overlap_tokens))
            else:
                sents.append(s)
        for sent in sents:
            st = count_tokens(sent)
            if parts and tok + st > chunk_tokens:
                flush(overlap=True)
            if parts and parts[-1]["type"] == "prose":
                parts[-1]["sents"].append((page, sent))
            else:
                parts.append({"type": "prose", "sents": [(page, sent)]})
            tok += st
            pages.add(page)

    def add_table(el: dict):
        nonlocal tok, pages
        rows = el["rows"]
        ncols = len(el["header"]) if el["header"] else (max((len(r) for r in rows), default=0))
        htok = count_tokens(_md_header_block(el["header"], ncols))
        tpages = el["pages"]

        if not rows:
            if parts and tok + htok > chunk_tokens:
                flush(overlap=True)
            parts.append({"type": "table", "table_id": el["table_id"], "header": el["header"], "rows": [], "pages": tpages})
            tok += htok
            pages.update(tpages)
            return

        i, part, rowcount = 0, None, 0
        while i < len(rows):
            rt = count_tokens(_md_row_line(rows[i], ncols))
            if part is None:
                if parts and tok + htok + rt > chunk_tokens:
                    flush(overlap=True)   # close the (prose) chunk before starting the table
                part = {"type": "table", "table_id": el["table_id"], "header": el["header"], "rows": [], "pages": tpages}
                parts.append(part)
                tok += htok
                pages.update(tpages)
                rowcount = 0
            over_tokens = part["rows"] and tok + rt > chunk_tokens
            over_rows = part["rows"] and table_max_rows > 0 and rowcount >= table_max_rows
            if over_tokens or over_rows:
                flush(overlap=False)      # mid-table split: header repeats in next chunk
                part = None
                continue
            part["rows"].append(rows[i])
            tok += rt
            rowcount += 1
            i += 1

    for el in elements:
        if el["kind"] == "prose":
            add_prose(el["page"], el["text"])
        else:
            add_table(el)
    flush(overlap=False)

    # Number the pieces of each (possibly split) table: part i/total.
    totals: dict = {}
    for rec in records:
        for p in rec["parts"]:
            if p["type"] == "table":
                totals[p["table_id"]] = totals.get(p["table_id"], 0) + 1
    seen: dict = {}
    for rec in records:
        for p in rec["parts"]:
            if p["type"] == "table":
                seen[p["table_id"]] = seen.get(p["table_id"], 0) + 1
                p["part"] = f"{seen[p['table_id']]}/{totals[p['table_id']]}"

    for i, rec in enumerate(records, start=1):
        rec["chunk_id"] = f"chunk_{i:04d}"
    return [{"chunk_id": r.pop("chunk_id"), **r} for r in records]


def write_chunks(pdf_path: Path, output_dir: Path,
                 chunk_tokens: int, overlap_tokens: int, table_max_rows: int) -> None:
    chunks = build_chunks(pdf_path, chunk_tokens, overlap_tokens, table_max_rows)

    counts = {"prose": 0, "table": 0, "mixed": 0}
    for c in chunks:
        counts[c["type"]] = counts.get(c["type"], 0) + 1

    payload = {
        "source_file": pdf_path.name,
        "params": {
            "chunk_tokens": chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "table_max_rows": table_max_rows,
            "exact_tokens": _USE_EXACT_TOKENS,
        },
        "total_chunks": len(chunks),
        "prose_only_chunks": counts["prose"],
        "table_only_chunks": counts["table"],
        "mixed_chunks": counts["mixed"],
        "chunks": chunks,
    }
    (output_dir / "chunks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Chunks written        : {(output_dir / 'chunks.json').resolve()}"
          f"  ({counts['prose']} prose + {counts['table']} table + {counts['mixed']} mixed = {len(chunks)})")


def write_prose(pdf_path: Path, output_dir: Path, fmt: str) -> None:
    """
    Write the normal (non-table) text per page to text/page_NNN.txt, and a
    combined all_text.txt. When fmt == 'json', also drop a text.json mapping
    page number -> prose.
    """
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    per_page = []
    combined = []
    for page_num, prose in extract_page_prose(str(pdf_path)):
        (text_dir / f"page_{page_num:03d}.txt").write_text(prose, encoding="utf-8")
        per_page.append({"page": page_num, "text": prose})
        if prose:
            combined.append(f"===== Page {page_num} =====\n{prose}")

    (output_dir / "all_text.txt").write_text("\n\n".join(combined), encoding="utf-8")

    if fmt == "json":
        (output_dir / "text.json").write_text(
            json.dumps(per_page, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Prose text (no tables): {text_dir.resolve()}")
    print(f"Combined prose        : {(output_dir / 'all_text.txt').resolve()}")


# ── Merged (stitched) mode ────────────────────────────────────────────────────

def run_merged(pdf_path: Path, output_dir: Optional[Path], fmt: str, opts: dict) -> None:
    tables = group_tables(str(pdf_path))
    json_tables = []

    for lt in tables:
        header_note = "with header" if lt.header is not None else "no header detected"
        print(
            f"\n=== Logical Table {lt.index + 1} | pages {lt.page_span} | "
            f"{len(lt.rows)} data rows x {lt.num_cols} cols | {header_note} ==="
        )
        if len(lt.pages) > 1:
            print(f"    (stitched across pages {lt.page_span}; header carried forward)")
        print_table(lt.all_rows)

        # Per-table CSV / tables.json are debug-only side artifacts.
        if output_dir and opts["debug"]:
            if fmt == "csv":
                save_table_csv(lt.all_rows, output_dir / f"table{lt.index + 1}_pages{lt.page_span}.csv")
            else:
                json_tables.append({
                    "table_index": lt.index + 1,
                    "pages": lt.pages,
                    "num_cols": lt.num_cols,
                    "header": lt.header,
                    "rows": lt.rows,
                })

    if output_dir and opts["debug"] and fmt == "json":
        with open(output_dir / "tables.json", "w", encoding="utf-8") as f:
            json.dump(json_tables, f, indent=2, ensure_ascii=False)

    print(f"\nTotal logical tables: {len(tables)}")
    if output_dir:
        _write_side_artifacts(pdf_path, output_dir, fmt, opts)


# ── Raw per-page mode (old behavior) ──────────────────────────────────────────

def run_raw(pdf_path: Path, output_dir: Optional[Path], fmt: str, opts: dict) -> None:
    json_tables = []
    total = 0

    for page_num, table_idx, table in extract_page_tables(str(pdf_path)):
        total += 1
        print(f"\n=== Page {page_num}, Table {table_idx} ({len(table)} rows) ===")
        print_table(table)

        # Per-table CSV / tables.json are debug-only side artifacts.
        if output_dir and opts["debug"]:
            if fmt == "csv":
                save_table_csv(table, output_dir / f"page{page_num}_table{table_idx}.csv")
            else:
                json_tables.append({"page": page_num, "table_index": table_idx, "rows": table})

    if output_dir and opts["debug"] and fmt == "json":
        with open(output_dir / "tables.json", "w", encoding="utf-8") as f:
            json.dump(json_tables, f, indent=2, ensure_ascii=False)

    print(f"\nTotal tables found: {total}")
    if output_dir:
        _write_side_artifacts(pdf_path, output_dir, fmt, opts)


def _write_side_artifacts(pdf_path: Path, output_dir: Path, fmt: str, opts: dict) -> None:
    """
    chunks.json is always written (the production artifact). The per-page prose
    text files are debug-only.
    """
    if opts["debug"]:
        write_prose(pdf_path, output_dir, fmt)
    write_chunks(
        pdf_path, output_dir,
        chunk_tokens=opts["chunk_tokens"],
        overlap_tokens=opts["overlap_tokens"],
        table_max_rows=opts["table_max_rows"],
    )


def main():
    parser = argparse.ArgumentParser(description="Extract tables (and non-table prose) from a PDF using pdfplumber")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default=None, help="If set, saves extracted tables + prose here")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Disable multi-page stitching; emit raw per-page tables (old behavior)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Also write the inspectable side artifacts (per-table CSV/tables.json, per-page prose text/ + all_text.txt). "
             "Without it, chunks.json is the ONLY output. Overrides the module DEBUG constant when passed.",
    )
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS,
                        help="Common token budget for every chunk — prose and tables share it (rule: ~300-500)")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help="Overlap in tokens carried between chunks when a boundary falls in prose (rule: ~50)")
    parser.add_argument("--table-max-rows", type=int, default=DEFAULT_TABLE_MAX_ROWS,
                        help="Optional hard row cap per table piece (0 = token-driven only)")
    parser.add_argument("--exact-tokens", action="store_true",
                        help="Count tokens with the BAAI/bge-m3 tokenizer (matches the main pipeline) instead of a fast heuristic")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    init_tokenizer(args.exact_tokens)

    opts = {
        "debug": DEBUG or args.debug,     # module constant, or --debug for this run
        "chunk_tokens": args.chunk_tokens,
        "overlap_tokens": args.chunk_overlap,
        "table_max_rows": args.table_max_rows,
    }

    if args.no_merge:
        run_raw(pdf_path, output_dir, args.format, opts)
    else:
        run_merged(pdf_path, output_dir, args.format, opts)


if __name__ == "__main__":
    main()
