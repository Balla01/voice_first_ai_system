"""
Docling-based page-by-page PDF extraction.

Docling runs each page through its layout model + the TableFormer structure
model, so paragraphs and tables come back as separate structured items
(each tagged with its own page number) instead of one flattened text stream.
That means a table's cells never bleed into the surrounding prose, and vice
versa — see miscellaneous/docling_table_extractor.py for the standalone CLI
version this was adapted from, including notes on TableFormer's accuracy
limits on long tables that span many pages.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import SectionHeaderItem, TableItem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constants import DOCLING_TABLE_MODE


@dataclass
class TableBlock:
    page: int
    markdown: str
    rows: int
    cols: int


@dataclass
class PageContent:
    page: int
    section: str = ""
    paragraphs: List[str] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Plain pipe-table renderer so we don't pull in the optional `tabulate` dep."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


class DoclingExtractor:
    """
    Wraps one DocumentConverter so the layout + TableFormer models are loaded
    once and reused across every PDF/batch — mirrors how PDFEmbeddingPipeline
    loads its SentenceTransformer once in __init__.
    """

    def __init__(self, table_mode: str = DOCLING_TABLE_MODE, do_ocr: bool = False):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
        )
        pipeline_options.table_structure_options.do_cell_matching = True
        # Docling defaults do_ocr=True; every product PDF in this corpus is
        # digital (verified: pypdfium2 extracts real text, no scanned pages),
        # so OCR only adds model downloads + per-page latency for no benefit.
        pipeline_options.do_ocr = do_ocr

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def extract_pages(
        self,
        pdf_path: str,
        page_range: Optional[Tuple[int, int]] = None,
        initial_section: str = "",
    ) -> Tuple[List[PageContent], str]:
        """
        Convert (optionally just page_range of) pdf_path and return
        (pages, ending_section) — one PageContent per page that has content,
        in reading order.

        `initial_section` / `ending_section` let the caller carry the current
        section heading across batches (each batch is a separate convert()
        call covering a different page_range of the same PDF, processed in
        order — see dump_pipeline.py), so a heading seen on page 3 still
        tags a table on page 5 with no heading of its own.
        """
        convert_kwargs = {"page_range": page_range} if page_range else {}
        result = self._converter.convert(pdf_path, **convert_kwargs)
        doc = result.document

        pages: Dict[int, PageContent] = {}
        current_section = initial_section
        for item, _level in doc.iterate_items():
            pno = item.prov[0].page_no if item.prov else None
            if pno is None:
                continue
            page = pages.get(pno)
            if page is None:
                page = PageContent(page=pno, section=current_section)
                pages[pno] = page

            if isinstance(item, SectionHeaderItem):
                current_section = item.text.strip()
                page.paragraphs.append(item.text.strip())
            elif isinstance(item, TableItem):
                df = item.export_to_dataframe(doc=doc)
                page.tables.append(TableBlock(
                    page=pno,
                    markdown=_dataframe_to_markdown(df),
                    rows=int(df.shape[0]),
                    cols=int(df.shape[1]),
                ))
            elif hasattr(item, "text") and item.text and item.text.strip():
                page.paragraphs.append(item.text.strip())

        ordered_pages = [pages[p] for p in sorted(pages)]
        return ordered_pages, current_section
