import sys
import os
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pypdfium2 as pdfium

from qmd.data.data_reader_writer.filebase import FileBasedDataWriter
from qmd.utils.pdf_classify import classify
from qmd.utils.pdf_image_tools import load_images_from_pdf_doc
from qmd.utils.enum_class import ImageType, MakeMode
from qmd.utils.config_reader import get_device
from qmd.utils.model_utils import get_vram, clean_memory
from qmd.utils.pdfium_guard import open_pdfium_document, get_pdfium_document_page_count, close_pdfium_document
from qmd.backend.pipeline.pipeline_analyze import ModelSingleton, batch_image_analyze
from qmd.backend.pipeline.model_json_to_middle_json import init_middle_json, append_batch_results_to_middle_json, finalize_middle_json
from qmd.backend.pipeline.pipeline_middle_json_mkcontent import union_make


def convert(pdf_path: str, output_dir: str, lang: str = "en", parse_method: str = "auto"):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    stem = pdf_path.stem

    image_dir = output_dir / stem / "auto" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_writer = FileBasedDataWriter(str(image_dir))

    pdf_bytes = open(pdf_path, "rb").read()

    if parse_method == "auto":
        ocr_enable = classify(pdf_bytes) == "ocr"
    else:
        ocr_enable = parse_method == "ocr"

    ModelSingleton().get_model(lang=lang, formula_enable=True, table_enable=True)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    page_count = get_pdfium_document_page_count(pdf_doc)
    images_list = load_images_from_pdf_doc(
        pdf_doc, start_page_id=0, end_page_id=page_count - 1,
        image_type=ImageType.PIL, pdf_bytes=pdf_bytes,
    )

    batch_results = batch_image_analyze(
        [(img["img_pil"], ocr_enable, lang) for img in images_list],
        formula_enable=True, table_enable=True,
    )

    middle_json = init_middle_json()
    append_batch_results_to_middle_json(
        middle_json, batch_results, images_list, pdf_doc, image_writer,
        page_start_index=0, ocr_enable=ocr_enable,
    )
    close_pdfium_document(pdf_doc)
    clean_memory(get_device())

    finalize_middle_json(middle_json["pdf_info"], lang=lang, ocr_enable=ocr_enable)
    markdown = union_make(middle_json["pdf_info"], MakeMode.MM_MD, "images")

    md_dir = output_dir / stem / "auto"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_file = md_dir / f"{stem}.md"
    md_file.write_text(markdown, encoding="utf-8")

    print(f"[OK] → {md_file}")
    return str(md_file)


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python run.py <pdf_path> [output_dir]")
        sys.exit(1)

    input_path = Path(args[0])
    output_dir = Path(args[1]) if len(args) > 1 else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_path.glob("*.pdf")) if input_path.is_dir() else [input_path]
    for pdf in pdfs:
        convert(str(pdf), str(output_dir))


if __name__ == "__main__":
    main()
