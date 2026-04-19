"""
MinerU pipeline - unboxed.

Steps:
  1. Load PDF bytes
  2. Classify: text-based or OCR
  3. Load all models (layout, OCR, formula, table)
  4. Render pages to PIL images
  5. Run batch inference (layout detection + OCR + formula + table)
  6. Build middle JSON (structured page blocks)
  7. Finalize (post-OCR, formula numbering, paragraph splitting)
  8. Generate markdown
  9. Write output files

Usage:
    python mineru_pipeline.py <pdf_path> [output_dir]
"""

import sys
import os
from pathlib import Path

import pypdfium2 as pdfium

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import load_images_from_pdf_doc
from mineru.utils.enum_class import ImageType, MakeMode
from mineru.utils.config_reader import get_device
from mineru.utils.model_utils import get_vram, clean_memory
from mineru.utils.pdfium_guard import (
    open_pdfium_document,
    get_pdfium_document_page_count,
    close_pdfium_document,
)
from mineru.backend.pipeline.pipeline_analyze import ModelSingleton, batch_image_analyze
from mineru.backend.pipeline.model_json_to_middle_json import (
    init_middle_json,
    append_batch_results_to_middle_json,
    finalize_middle_json,
)
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make


# ── Step 1: Load PDF ──────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> bytes:
    return pdf_path.read_bytes()


# ── Step 2: Classify ──────────────────────────────────────────────────────────

def get_ocr_enable(pdf_bytes: bytes, parse_method: str = "auto") -> bool:
    """
    "auto"  → classify the PDF; if scanned/image-based, enable OCR
    "ocr"   → always use OCR
    "txt"   → never use OCR (trust embedded text layer)
    """
    if parse_method == "auto":
        return classify(pdf_bytes) == "ocr"
    return parse_method == "ocr"


# ── Step 3: Load models ───────────────────────────────────────────────────────

def load_models(lang: str = "en", formula_enable: bool = True, table_enable: bool = True):
    """
    Returns the shared MineruPipelineModel singleton.
    Models loaded:
      - PPDocLayoutV2LayoutModel  (layout detection)
      - UnimernetModel            (formula recognition)
      - PytorchPaddleOCR          (text OCR det + rec)
      - PaddleTableClsModel       (table classification: wired vs wireless)
      - UnetTableModel            (wired table → HTML)
      - PaddleTableModel          (wireless table → HTML)
    """
    model_manager = ModelSingleton()
    model = model_manager.get_model(
        lang=lang,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )
    return model_manager, model


# ── Step 4: Render pages ──────────────────────────────────────────────────────

def render_pages(pdf_bytes: bytes, start_page: int = 0, end_page: int = None):
    """
    Open PDF with pypdfium2 and render pages to PIL images.
    Returns (pdf_doc, images_list) where each image_dict has:
      - img_pil: PIL.Image (RGB)
      - scale: float (72dpi → 144dpi = 2.0)
    """
    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    page_count = get_pdfium_document_page_count(pdf_doc)
    if end_page is None:
        end_page = page_count - 1
    images_list = load_images_from_pdf_doc(
        pdf_doc,
        start_page_id=start_page,
        end_page_id=end_page,
        image_type=ImageType.PIL,
        pdf_bytes=pdf_bytes,
    )
    return pdf_doc, images_list, page_count


# ── Step 5: Batch inference ───────────────────────────────────────────────────

def run_batch_inference(
    images_list: list,
    ocr_enable: bool,
    lang: str,
    formula_enable: bool = True,
    table_enable: bool = True,
) -> list:
    """
    Run all models on a batch of page images.
    Returns a list (one per page) of layout detection results.
    Each result is a list of dicts: {bbox, label, score, latex/html/text, ...}

    Models invoked (in order per page):
      1. Layout detection    → bounding boxes + labels
      2. Formula recognition → latex string for each formula bbox
      3. Table classification + OCR → HTML for each table bbox
      4. Text OCR detection + recognition → text for each text region
    """
    images_with_extra_info = [
        (image_dict["img_pil"], ocr_enable, lang)
        for image_dict in images_list
    ]
    batch_results = batch_image_analyze(
        images_with_extra_info,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )
    return batch_results


# ── Step 6: Build middle JSON ─────────────────────────────────────────────────

def build_middle_json(
    batch_results: list,
    images_list: list,
    pdf_doc,
    image_writer,
    page_start: int = 0,
    ocr_enable: bool = False,
) -> dict:
    """
    Convert raw model outputs into structured page blocks (middle JSON).
    Each page gets:
      - preproc_blocks: list of content blocks (text, title, table, image, equation, ...)
      - discarded_blocks: headers, footers, footnotes
      - page_idx, page_size

    Also cuts and saves image/table/formula crops to image_writer.
    """
    middle_json = init_middle_json()
    append_batch_results_to_middle_json(
        middle_json,
        batch_results,
        images_list,
        pdf_doc,
        image_writer,
        page_start_index=page_start,
        ocr_enable=ocr_enable,
    )
    return middle_json


# ── Step 7: Finalize ──────────────────────────────────────────────────────────

def finalize(middle_json: dict, lang: str, ocr_enable: bool):
    """
    Document-level post-processing:
      - Post-OCR recognition for remaining image crops
      - Formula number tags (e.g., \\tag{(1)})
      - Paragraph splitting (group lines into paragraphs)
      - Cross-page table merging
    Modifies middle_json["pdf_info"] in place.
    """
    finalize_middle_json(middle_json["pdf_info"], lang=lang, ocr_enable=ocr_enable)


# ── Step 8: Generate markdown ─────────────────────────────────────────────────

def generate_markdown(middle_json: dict, image_dir_name: str = "images") -> str:
    """
    Convert structured page blocks into markdown text.
    Block type → markdown:
      TEXT / ABSTRACT / LIST  →  paragraph
      TITLE (level 1/2/3)     →  # / ## / ###
      INTERLINE_EQUATION       →  $$latex$$
      INLINE_EQUATION          →  $latex$
      TABLE                    →  HTML table
      IMAGE / CHART            →  ![](image_path)
      CODE                     →  ```code```
    """
    return union_make(middle_json["pdf_info"], MakeMode.MM_MD, image_dir_name)


# ── Step 9: Write output ──────────────────────────────────────────────────────

def write_output(markdown: str, stem: str, output_dir: Path):
    """Write markdown file. Images were already written by image_writer during step 6."""
    md_dir = output_dir / stem / "auto"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_file = md_dir / f"{stem}.md"
    md_file.write_text(markdown, encoding="utf-8")
    return md_file


# ── Full pipeline ─────────────────────────────────────────────────────────────

def convert(pdf_path: Path, output_dir: Path, lang: str = "en", parse_method: str = "auto"):
    stem = pdf_path.stem
    image_dir = output_dir / stem / "auto" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_writer = FileBasedDataWriter(str(image_dir))

    print(f"[1/7] Loading PDF: {pdf_path.name}")
    pdf_bytes = load_pdf(pdf_path)

    print(f"[2/7] Classifying PDF...")
    ocr_enable = get_ocr_enable(pdf_bytes, parse_method)
    print(f"       → OCR mode: {ocr_enable}")

    print(f"[3/7] Loading models (lang={lang})...")
    model_manager, _ = load_models(lang=lang, formula_enable=True, table_enable=True)

    print(f"[4/7] Rendering pages...")
    pdf_doc, images_list, page_count = render_pages(pdf_bytes)
    print(f"       → {page_count} pages")

    print(f"[5/7] Running batch inference...")
    batch_results = run_batch_inference(
        images_list, ocr_enable, lang, formula_enable=True, table_enable=True
    )

    print(f"[6/7] Building middle JSON...")
    middle_json = build_middle_json(
        batch_results, images_list, pdf_doc, image_writer, ocr_enable=ocr_enable
    )
    close_pdfium_document(pdf_doc)
    clean_memory(get_device())

    print(f"[7/7] Finalizing and generating markdown...")
    finalize(middle_json, lang=lang, ocr_enable=ocr_enable)
    markdown = generate_markdown(middle_json, image_dir_name="images")

    md_file = write_output(markdown, stem, output_dir)
    print(f"[OK]  {pdf_path.name} → {md_file}")
    return md_file


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    input_path = Path(args[0])
    output_dir = Path(args[1]) if len(args) > 1 else input_path.parent

    if input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
    elif input_path.is_file():
        pdfs = [input_path]
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        convert(pdf, output_dir)


if __name__ == "__main__":
    main()
