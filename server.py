import sys
import os
import tempfile
import urllib.request
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pypdfium2 as pdfium
from fastapi import FastAPI
from pydantic import BaseModel

from qmd.data.data_reader_writer.filebase import FileBasedDataWriter
from qmd.utils.pdf_classify import classify
from qmd.utils.pdf_image_tools import load_images_from_pdf_doc
from qmd.utils.enum_class import ImageType, MakeMode
from qmd.utils.config_reader import get_device
from qmd.utils.model_utils import clean_memory
from qmd.utils.pdfium_guard import open_pdfium_document, get_pdfium_document_page_count, close_pdfium_document
from qmd.backend.pipeline.pipeline_analyze import ModelSingleton, batch_image_analyze
from qmd.backend.pipeline.model_json_to_middle_json import init_middle_json, append_batch_results_to_middle_json, finalize_middle_json
from qmd.backend.pipeline.pipeline_middle_json_mkcontent import union_make

OUTPUT_DIR = Path(_HERE) / "raw"
app = FastAPI()


@app.on_event("startup")
def load_models():
    ModelSingleton().get_model(lang="en", formula_enable=True, table_enable=True)


class ConvertRequest(BaseModel):
    url: str
    lang: str = "en"
    parse_method: str = "auto"


@app.post("/convert")
def convert(req: ConvertRequest):
    pdf_path = Path(req.url)
    if pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
        stem = pdf_path.stem
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            urllib.request.urlretrieve(req.url, tmp.name)
            pdf_bytes = open(tmp.name, "rb").read()
            stem = Path(req.url).stem
        os.unlink(tmp.name)

    if req.parse_method == "auto":
        ocr_enable = classify(pdf_bytes) == "ocr"
    else:
        ocr_enable = req.parse_method == "ocr"

    image_dir = OUTPUT_DIR / stem / "auto" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_writer = FileBasedDataWriter(str(image_dir))

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    page_count = get_pdfium_document_page_count(pdf_doc)
    images_list = load_images_from_pdf_doc(
        pdf_doc, start_page_id=0, end_page_id=page_count - 1,
        image_type=ImageType.PIL, pdf_bytes=pdf_bytes,
    )

    batch_results = batch_image_analyze(
        [(img["img_pil"], ocr_enable, req.lang) for img in images_list],
        formula_enable=True, table_enable=True,
    )

    middle_json = init_middle_json()
    append_batch_results_to_middle_json(
        middle_json, batch_results, images_list, pdf_doc, image_writer,
        page_start_index=0, ocr_enable=ocr_enable,
    )
    close_pdfium_document(pdf_doc)
    clean_memory(get_device())

    finalize_middle_json(middle_json["pdf_info"], lang=req.lang, ocr_enable=ocr_enable)
    markdown = union_make(middle_json["pdf_info"], MakeMode.MM_MD, "images")

    md_file = OUTPUT_DIR / stem / "auto" / f"{stem}.md"
    md_file.write_text(markdown, encoding="utf-8")

    return {"markdown_file": str(md_file), "pages": page_count}
