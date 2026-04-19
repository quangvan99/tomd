"""
MinerU pipeline - fully self-contained, no installed qmd package needed.

All source code lives in ./qmd/  (copied from qmd site-packages)
All model weights live in ./models/ (copied from HuggingFace cache)

Steps:
  1. Load PDF bytes
  2. Classify: text-based or OCR
  3. Load all models (layout, OCR, formula, table) from ./models/
  4. Render pages to PIL images
  5. Run batch inference (layout detection + OCR + formula + table)
  6. Build middle JSON (structured page blocks)
  7. Finalize (post-OCR, formula numbering, paragraph splitting)
  8. Generate markdown
  9. Write output files

Usage:
    python qmd_pipeline.py <pdf_path> [output_dir]
"""

import sys
import os

# Use local qmd/ source, not the installed package
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pypdfium2 as pdfium

from qmd.data.data_reader_writer.filebase import FileBasedDataWriter
from qmd.utils.pdf_classify import classify
from qmd.utils.pdf_image_tools import load_images_from_pdf_doc
from qmd.utils.enum_class import ImageType, MakeMode
from qmd.utils.config_reader import get_device
from qmd.utils.model_utils import get_vram, clean_memory
from qmd.utils.pdfium_guard import (
    open_pdfium_document,
    get_pdfium_document_page_count,
    close_pdfium_document,
)
from qmd.backend.pipeline.pipeline_analyze import ModelSingleton, batch_image_analyze
from qmd.backend.pipeline.model_json_to_middle_json import (
    init_middle_json,
    append_batch_results_to_middle_json,
    finalize_middle_json,
)
from qmd.backend.pipeline.pipeline_middle_json_mkcontent import union_make


# ── Step 1: Load PDF ──────────────────────────────────────────────────────────

def load_pdf(pdf_path) -> bytes:
    return open(pdf_path, "rb").read()


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
    Load all models from ./models/ (no download).

    Models loaded:
      - PPDocLayoutV2LayoutModel    models/Layout/PP-DocLayoutV2/
      - UnimernetModel              models/MFR/unimernet_hf_small_2503/
      - PytorchPaddleOCR            models/OCR/paddleocr_torch/
      - PaddleTableClsModel         models/TabCls/paddle_table_cls/*.onnx
      - UnetTableModel              models/TabRec/UnetStructure/unet.onnx
      - PaddleTableModel            models/TabRec/SlanetPlus/slanet-plus.onnx
    """
    model_manager = ModelSingleton()
    model_manager.get_model(
        lang=lang,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )
    return model_manager


# ── Step 4: Render pages ──────────────────────────────────────────────────────

def render_pages(pdf_bytes: bytes, start_page: int = 0, end_page: int = None):
    """
    Open PDF with pypdfium2 and render all pages to PIL images (144 DPI).
    Returns (pdf_doc, images_list, page_count).
    Each entry in images_list: {"img_pil": PIL.Image, "scale": float}
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
    Run all models on all page images in one batch.

    Order of operations per page:
      1. Layout detection      → bboxes + labels (text/table/image/formula/...)
      2. Formula recognition   → LaTeX string for each formula bbox
      3. Table classification  → wired vs wireless
      4. Table OCR + structure → HTML per table
      5. Text OCR det + rec    → text content per text region

    Returns list (one per page) of layout detection dicts:
      {bbox, label, score, latex (formulas), html (tables), text (OCR), ...}
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
    Convert raw model outputs to structured block tree (middle JSON).

    For each page builds:
      preproc_blocks: [
        {type, bbox, lines: [{bbox, spans: [{type, content, score, image_path, bbox}]}]}
      ]
      discarded_blocks: headers, footers, footnotes

    Also crops and saves image/table/formula regions to image_writer.
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
    Document-level post-processing (modifies middle_json in place):
      - Post-OCR recognition for any remaining np_img crops
      - Formula number tags: attach \\tag{(n)} to adjacent equations
      - Paragraph splitting: merge lines into reading-order paragraphs
      - Cross-page table merging
    """
    finalize_middle_json(middle_json["pdf_info"], lang=lang, ocr_enable=ocr_enable)


# ── Step 8: Generate markdown ─────────────────────────────────────────────────

def generate_markdown(middle_json: dict, image_dir_name: str = "images") -> str:
    """
    Walk the block tree and emit markdown.

    Block type → markdown output:
      TEXT / ABSTRACT / LIST       → plain paragraph
      TITLE level 1/2/3+           → # / ## / ###
      INTERLINE_EQUATION            → $$latex$$
      INLINE_EQUATION               → $latex$
      TABLE                         → <html table>
      IMAGE / CHART                 → ![](images/xxx.jpg)
      CODE                          → ```...```
    """
    return union_make(middle_json["pdf_info"], MakeMode.MM_MD, image_dir_name)


# ── Step 9: Write output ──────────────────────────────────────────────────────

def write_output(markdown: str, stem: str, output_dir) -> str:
    """Write .md file. Images already written by image_writer in step 6."""
    from pathlib import Path
    md_dir = Path(output_dir) / stem / "auto"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_file = md_dir / f"{stem}.md"
    md_file.write_text(markdown, encoding="utf-8")
    return str(md_file)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def convert(pdf_path: str, output_dir: str, lang: str = "en", parse_method: str = "auto"):
    from pathlib import Path
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    stem = pdf_path.stem

    image_dir = output_dir / stem / "auto" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_writer = FileBasedDataWriter(str(image_dir))

    print(f"[1/7] Loading PDF:          {pdf_path.name}")
    pdf_bytes = load_pdf(pdf_path)

    print(f"[2/7] Classifying PDF...")
    ocr_enable = get_ocr_enable(pdf_bytes, parse_method)
    print(f"       → ocr_enable={ocr_enable}")

    print(f"[3/7] Loading models (lang={lang}, models_root=./models/)...")
    load_models(lang=lang, formula_enable=True, table_enable=True)

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

    print(f"[7/7] Finalizing + generating markdown...")
    finalize(middle_json, lang=lang, ocr_enable=ocr_enable)
    markdown = generate_markdown(middle_json, image_dir_name="images")

    md_file = write_output(markdown, stem, output_dir)
    print(f"[OK]  → {md_file}")
    return md_file


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    from pathlib import Path
    input_path = Path(args[0])
    output_dir = Path(args[1]) if len(args) > 1 else input_path.parent

    pdfs = sorted(input_path.glob("*.pdf")) if input_path.is_dir() else [input_path]
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        convert(str(pdf), str(output_dir))


if __name__ == "__main__":
    main()
