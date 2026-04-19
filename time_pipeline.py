"""Pipeline timing profiler — wraps each major step with perf_counter."""
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import pypdfium2 as pdfium

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

# ── patch BatchAnalyze.__call__ to emit per-step timings ──────────────────────
import qmd.backend.pipeline.batch_analyze as _ba
import numpy as np
from collections import defaultdict
from tqdm import tqdm

_orig_call = _ba.BatchAnalyze.__call__

def _timed_call(self, images_with_extra_info):
    if not images_with_extra_info:
        return []

    T = OrderedDict()

    self.model = self.model_manager.get_model(
        lang=None,
        formula_enable=self.formula_enable,
        table_enable=self.table_enable,
    )
    atom_model_manager = _ba.AtomModelSingleton()
    pil_images  = [img for img, _, _ in images_with_extra_info]
    np_images   = [np.asarray(img) for img, _, _ in images_with_extra_info]

    # ── Layout ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    images_layout_res = self.model.layout_model.batch_predict(
        pil_images, batch_size=min(8, self.batch_ratio * _ba.LAYOUT_BASE_BATCH_SIZE)
    )
    _ba.clean_vram(self.model.device, vram_threshold=8)
    T["1. Layout detect"] = time.perf_counter() - t0

    # ── Formula recognition ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    if self.formula_enable:
        images_mfd_res = []
        for layout_res in images_layout_res:
            page_formula_res = []
            for res in layout_res:
                if res.get("label") in ["display_formula", "inline_formula"]:
                    res.setdefault("latex", "")
                    page_formula_res.append(res)
            images_mfd_res.append(page_formula_res)

        images_formula_list = self.model.mfr_model.batch_predict(
            images_mfd_res, np_images,
            batch_size=self.batch_ratio * _ba.MFR_BASE_BATCH_SIZE,
        )
        for image_index in range(len(np_images)):
            for formula_res, formula_with_latex in zip(
                images_mfd_res[image_index], images_formula_list[image_index]
            ):
                formula_res["latex"] = formula_with_latex.get("latex", "")
        _ba.clean_vram(self.model.device, vram_threshold=8)
    else:
        for layout_res in images_layout_res:
            layout_res[:] = [r for r in layout_res if r.get("label") != "inline_formula"]
    T["2. Formula (MFR)"] = time.perf_counter() - t0

    # ── Build ocr/table res lists ────────────────────────────────────────────
    t0 = time.perf_counter()
    ocr_res_list_all_page   = []
    table_res_list_all_page = []
    for index in range(len(np_images)):
        _, ocr_enable, _lang = images_with_extra_info[index]
        layout_res = images_layout_res[index]
        np_img     = np_images[index]
        table_inline_objects = (
            self._extract_table_inline_objects(layout_res, np_img, self.formula_enable)
            if self.table_enable else {}
        )
        ocr_res_list, table_res_list, single_page_mfdetrec_res = (
            _ba.get_res_list_from_layout_res(layout_res)
        )
        ocr_res_list_all_page.append({
            "ocr_res_list": ocr_res_list, "lang": _lang,
            "ocr_enable": ocr_enable, "np_img": np_img,
            "single_page_mfdetrec_res": single_page_mfdetrec_res,
            "layout_res": layout_res,
        })
        for table_res in table_res_list:
            def get_crop_table_img(scale, _tr=table_res, _ni=np_img):
                bbox = _ba.normalize_to_int_bbox(
                    [float(v) / float(scale) for v in _tr["bbox"]]
                )
                if bbox is None:
                    return _ni[0:0, 0:0]
                return _ba.get_crop_np_img(bbox, _ni, scale=scale)
            wireless_table_img = get_crop_table_img(1)
            wired_table_img    = get_crop_table_img(10/3)
            table_page_bbox    = _ba.normalize_to_int_bbox(
                table_res.get("bbox"), image_size=np_img.shape[:2]
            ) or [0, 0, 0, 0]
            table_res_list_all_page.append({
                "table_res": table_res, "lang": _lang,
                "table_img": wireless_table_img,
                "wired_table_img": wired_table_img,
                "table_page_bbox": table_page_bbox,
                "table_inline_objects": table_inline_objects.get(id(table_res), []),
            })
    T["3. Build res lists"] = time.perf_counter() - t0

    # ── Table pipeline ───────────────────────────────────────────────────────
    if self.table_enable:
        # orientation cls
        t0 = time.perf_counter()
        img_ori_model = atom_model_manager.get_atom_model(
            atom_model_name=_ba.AtomicModel.ImgOrientationCls,
        )
        try:
            if self.table_ori_cls_batch_enabled:
                img_ori_model.batch_predict(
                    table_res_list_all_page,
                    det_batch_size=self.batch_ratio * _ba.OCR_DET_BASE_BATCH_SIZE,
                    batch_size=_ba.TABLE_ORI_CLS_BATCH_SIZE,
                )
            else:
                for tr in table_res_list_all_page:
                    lbl = img_ori_model.predict(tr["table_img"])
                    img_ori_model.img_rotate(tr, lbl)
        except Exception as e:
            from loguru import logger; logger.warning(f"Orientation cls failed: {e}")
        T["4a. Table orientation cls"] = time.perf_counter() - t0

        # table cls
        t0 = time.perf_counter()
        table_cls_model = atom_model_manager.get_atom_model(
            atom_model_name=_ba.AtomicModel.TableCls,
        )
        try:
            table_cls_model.batch_predict(
                table_res_list_all_page,
                batch_size=_ba.TABLE_Wired_Wireless_CLS_BATCH_SIZE,
            )
        except Exception as e:
            from loguru import logger; logger.warning(f"Table cls failed: {e}")
        T["4b. Table cls (wired/wireless)"] = time.perf_counter() - t0

        # table OCR det (batched)
        t0 = time.perf_counter()
        rec_img_lang_group = defaultdict(list)
        det_ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name=_ba.AtomicModel.OCR,
            det_db_box_thresh=0.5, det_db_unclip_ratio=1.6, enable_merge_det_boxes=False,
        )
        table_det_data = []
        for index, tr in enumerate(table_res_list_all_page):
            bgr_image = _ba.cv2.cvtColor(tr["table_img"], _ba.cv2.COLOR_RGB2BGR)
            tio = (tr.get("table_inline_objects", []) if self._table_supports_inline_objects(tr) else [])
            inline_mask = [{"bbox": o["table_rel_mask_bbox"]} for o in tio]
            formula_mask = [{"bbox": o["table_rel_mask_bbox"]} for o in tio if o["kind"] == "formula"]
            det_img = (self._apply_mask_boxes_to_image(bgr_image, inline_mask) if inline_mask else bgr_image)
            table_det_data.append((index, bgr_image, det_img, formula_mask, tr["lang"]))

        STRIDE = 64
        res_groups = defaultdict(list)
        for item in table_det_data:
            h, w = item[2].shape[:2]
            th = ((h + STRIDE - 1) // STRIDE) * STRIDE
            tw = ((w + STRIDE - 1) // STRIDE) * STRIDE
            res_groups[(th, tw)].append(item)

        for (th, tw), grp in res_groups.items():
            batch_imgs = []
            for item in grp:
                img = item[2]; h, w = img.shape[:2]
                pad = np.ones((th, tw, 3), dtype=np.uint8) * 255
                pad[:h, :w] = img
                batch_imgs.append(pad)
            bsz = min(len(batch_imgs), self.batch_ratio * _ba.OCR_DET_BASE_BATCH_SIZE)
            results = det_ocr_engine.text_detector.batch_predict(batch_imgs, bsz)
            for item, (dt_boxes, _) in zip(grp, results):
                idx, bgr_img, _, fmask, ilang = item
                ocr_r = list(dt_boxes) if dt_boxes is not None else []
                if ocr_r and fmask:
                    ocr_r = _ba.update_det_boxes(ocr_r, fmask)
                if ocr_r:
                    ocr_r = _ba.sorted_boxes(ocr_r)
                for dt_box in ocr_r:
                    rec_img_lang_group[ilang].append({
                        "cropped_img": _ba.get_rotate_crop_image_for_text_rec(bgr_img, np.asarray(dt_box, dtype=np.float32)),
                        "dt_box": np.asarray(dt_box, dtype=np.float32),
                        "table_id": idx,
                    })
        T["4c. Table OCR det (batched)"] = time.perf_counter() - t0

        # table OCR rec
        t0 = time.perf_counter()
        for _lang, rec_list in rec_img_lang_group.items():
            if not rec_list:
                continue
            ocr_eng = atom_model_manager.get_atom_model(
                atom_model_name=_ba.AtomicModel.OCR,
                det_db_box_thresh=0.5, det_db_unclip_ratio=1.6,
                lang=_lang, enable_merge_det_boxes=False,
            )
            crops = [it["cropped_img"] for it in rec_list]
            ocr_res = ocr_eng.ocr(crops, det=False, tqdm_enable=True, tqdm_desc=f"Table-ocr rec {_lang}")[0]
            for img_dict, res in zip(rec_list, ocr_res):
                import html as _html
                entry = table_res_list_all_page[img_dict["table_id"]]
                row = [img_dict["dt_box"], _html.escape(res[0]), res[1]]
                entry.setdefault("ocr_result", []).append(row)
        T["4d. Table OCR rec"] = time.perf_counter() - t0

        # inline objects merge
        t0 = time.perf_counter()
        for tr in table_res_list_all_page:
            if not self._table_supports_inline_objects(tr):
                continue
            tio = tr.get("table_inline_objects", [])
            if not tio:
                continue
            ocr_r = tr.setdefault("ocr_result", [])
            for obj in tio:
                ocr_r.append([self._bbox_to_quad(obj["table_token_bbox"]), obj["content"], obj["score"]])
            self._sort_table_ocr_result(ocr_r)
        T["4e. Inline objects merge"] = time.perf_counter() - t0

        # wireless predict
        t0 = time.perf_counter()
        wireless_model = atom_model_manager.get_atom_model(atom_model_name=_ba.AtomicModel.WirelessTable)
        wireless_model.batch_predict(table_res_list_all_page)
        T["4f. Table wireless predict"] = time.perf_counter() - t0

        # wired predict
        t0 = time.perf_counter()
        wired_list = []
        for tr in table_res_list_all_page:
            if (
                (tr["table_res"]["cls_label"] == _ba.AtomicModel.WirelessTable and tr["table_res"]["cls_score"] < 0.9)
                or tr["table_res"]["cls_label"] == _ba.AtomicModel.WiredTable
            ):
                wired_list.append(tr)
            del tr["table_res"]["cls_label"]
            del tr["table_res"]["cls_score"]
        if wired_list:
            wired_lang_groups = defaultdict(list)
            for tr in wired_list:
                if tr.get("ocr_result"):
                    wired_lang_groups[tr["lang"]].append(tr)
            for lang, grp in wired_lang_groups.items():
                wired_model = atom_model_manager.get_atom_model(
                    atom_model_name=_ba.AtomicModel.WiredTable, lang=lang,
                )
                for tr in grp:
                    tr["table_res"]["html"] = wired_model.predict(
                        tr["wired_table_img"], tr["ocr_result"],
                        tr["table_res"].get("html", None),
                    )
        T["4g. Table wired predict"] = time.perf_counter() - t0

        # table HTML cleanup
        t0 = time.perf_counter()
        for tr in table_res_list_all_page:
            html_code = tr["table_res"].get("html", "") or ""
            if "<table>" in html_code and "</table>" in html_code:
                s = html_code.find("<table>")
                e = html_code.rfind("</table>") + len("</table>")
                tr["table_res"]["html"] = html_code[s:e]
        T["4h. Table HTML cleanup"] = time.perf_counter() - t0

    # ── Text OCR det ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if self.text_ocr_det_batch_enabled:
        all_crop_info = []
        for d in ocr_res_list_all_page:
            for res in d["ocr_res_list"]:
                new_img, useful_list = _ba.crop_img(res, d["np_img"], crop_paste_x=50, crop_paste_y=50)
                adj = _ba.get_adjusted_mfdetrec_res(d["single_page_mfdetrec_res"], useful_list)
                bgr = _ba.cv2.cvtColor(new_img, _ba.cv2.COLOR_RGB2BGR)
                det_img = self._get_masked_det_image(bgr, adj)
                all_crop_info.append((bgr, det_img, useful_list, d, adj, d["lang"]))

        lang_groups = defaultdict(list)
        for ci in all_crop_info:
            lang_groups[ci[5]].append(ci)

        for lang, lcrops in lang_groups.items():
            ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=_ba.AtomicModel.OCR, det_db_box_thresh=0.3, lang=lang,
            )
            STRIDE = 64
            res_grps = defaultdict(list)
            for ci in lcrops:
                h, w = ci[1].shape[:2]
                th = ((h + STRIDE - 1) // STRIDE) * STRIDE
                tw = ((w + STRIDE - 1) // STRIDE) * STRIDE
                res_grps[(th, tw)].append(ci)
            for (th, tw), grp in res_grps.items():
                batch_imgs = []
                for ci in grp:
                    img = ci[1]; h, w = img.shape[:2]
                    pad = np.ones((th, tw, 3), dtype=np.uint8) * 255
                    pad[:h, :w] = img
                    batch_imgs.append(pad)
                bsz = min(len(batch_imgs), self.batch_ratio * _ba.OCR_DET_BASE_BATCH_SIZE)
                results = ocr_model.text_detector.batch_predict(batch_imgs, bsz)
                for ci, (dt_boxes, _) in zip(grp, results):
                    bgr_img, _, useful_list, d, adj, _lang = ci
                    if dt_boxes is not None and len(dt_boxes) > 0:
                        sorted_b = _ba.sorted_boxes(dt_boxes)
                        merged   = _ba.merge_det_boxes(sorted_b) if sorted_b else []
                        final    = (_ba.update_det_boxes(merged, adj) if merged and adj else merged)
                        if final:
                            ocr_r = [b.tolist() if hasattr(b, "tolist") else b for b in final]
                            d["layout_res"].extend(
                                _ba.get_ocr_result_list(ocr_r, useful_list, d["ocr_enable"], bgr_img, _lang)
                            )
        _ba.clean_vram(self.model.device, vram_threshold=8)
    T["5. Text OCR det"] = time.perf_counter() - t0

    # ── Text OCR rec ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    need_ocr_by_lang = {}
    crop_by_lang     = {}
    for layout_res in images_layout_res:
        for item in layout_res:
            if not item.get("_need_ocr_rec"):
                continue
            if "np_img" in item and "lang" in item:
                lang = item["lang"]
                need_ocr_by_lang.setdefault(lang, []).append((layout_res, item))
                crop_by_lang.setdefault(lang, []).append(item["np_img"])
                item.pop("np_img", None); item.pop("lang", None); item.pop("_need_ocr_rec", None)

    for lang, crops in crop_by_lang.items():
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=_ba.AtomicModel.OCR, det_db_box_thresh=0.3, lang=lang,
        )
        ocr_res = ocr_model.ocr(crops, det=False, tqdm_enable=True)[0]
        items_to_remove = []
        for idx, (page_lr, item) in enumerate(need_ocr_by_lang[lang]):
            text, score = ocr_res[idx]
            item["text"]  = text
            item["score"] = float(f"{score:.3f}")
            if score < _ba.OcrConfidence.min_confidence:
                items_to_remove.append((page_lr, item))
            else:
                bb = item["bbox"]
                w, h = bb[2] - bb[0], bb[3] - bb[1]
                if (text in ['（204号','（20','（2','（2号','（20号','号','（204','(cid:)','(ci:)','(cd:1)','cd:)','c)','(cd:)','c','id:)',':)','√:)','√i:)','−i:)','−:','i:)']
                    and score < 0.8 and w < h):
                    items_to_remove.append((page_lr, item))
        for page_lr, item in items_to_remove:
            if item in page_lr:
                page_lr.remove(item)
    T["6. Text OCR rec"] = time.perf_counter() - t0

    # ── Seal OCR ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    seal_items = [
        (d, item)
        for d in ocr_res_list_all_page
        for item in d["layout_res"]
        if item.get("label") == "seal"
    ]
    seal_model = None
    for d, item in seal_items:
        np_img = d["np_img"]
        h, w   = np_img.shape[:2]
        item["text"] = ""
        bbox = _ba.normalize_to_int_bbox(item.get("bbox"), image_size=(h, w))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        crop = np_img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        if seal_model is None:
            seal_model = atom_model_manager.get_atom_model(
                atom_model_name=_ba.AtomicModel.OCR, lang="seal",
            )
        bgr = _ba.cv2.cvtColor(crop, _ba.cv2.COLOR_RGB2BGR)
        res = seal_model.ocr(bgr, det=True, rec=True)[0]
        if res:
            item["text"] = [r[1][0] for r in res if r and len(r) == 2 and r[1] and r[1][0]]
    T["7. Seal OCR"] = time.perf_counter() - t0

    # ── Prune empty OCR text blocks ───────────────────────────────────────────
    t0 = time.perf_counter()
    for d in ocr_res_list_all_page:
        self._prune_empty_ocr_text_blocks(d["layout_res"], d["ocr_enable"])
    T["8. Prune empty blocks"] = time.perf_counter() - t0

    # ── Print timing summary ─────────────────────────────────────────────────
    total_inner = sum(T.values())
    print("\n" + "═"*55)
    print(f"  BatchAnalyze internal timing  ({len(np_images)} pages)")
    print("═"*55)
    for name, t in T.items():
        bar = "█" * int(t / total_inner * 30)
        print(f"  {name:<35} {t:6.2f}s  {bar}")
    print("─"*55)
    print(f"  {'TOTAL (inner)':<35} {total_inner:6.2f}s")
    print("═"*55 + "\n")

    return images_layout_res

_ba.BatchAnalyze.__call__ = _timed_call
# ──────────────────────────────────────────────────────────────────────────────

def run(pdf_path: str, output_dir: str, lang: str = "en"):
    pdf_path   = Path(pdf_path)
    output_dir = Path(output_dir)
    stem       = pdf_path.stem

    image_dir = output_dir / stem / "auto" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_writer = FileBasedDataWriter(str(image_dir))

    pdf_bytes = open(pdf_path, "rb").read()

    T = OrderedDict()

    t0 = time.perf_counter()
    ocr_enable = classify(pdf_bytes) == "ocr"
    T["classify"] = time.perf_counter() - t0

    # pre-warm model (like service)
    t0 = time.perf_counter()
    ModelSingleton().get_model(lang=lang, formula_enable=True, table_enable=True)
    T["model warm (ignored)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pdf_doc     = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    page_count  = get_pdfium_document_page_count(pdf_doc)
    images_list = load_images_from_pdf_doc(
        pdf_doc, start_page_id=0, end_page_id=page_count - 1,
        image_type=ImageType.PIL, pdf_bytes=pdf_bytes,
    )
    T["PDF render"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch_results = batch_image_analyze(
        [(img["img_pil"], ocr_enable, lang) for img in images_list],
        formula_enable=True, table_enable=True,
    )
    T["batch_image_analyze (total)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    middle_json = init_middle_json()
    append_batch_results_to_middle_json(
        middle_json, batch_results, images_list, pdf_doc, image_writer,
        page_start_index=0, ocr_enable=ocr_enable,
    )
    T["append_batch → middle_json"] = time.perf_counter() - t0

    close_pdfium_document(pdf_doc)
    clean_memory(get_device())

    t0 = time.perf_counter()
    finalize_middle_json(middle_json["pdf_info"], lang=lang, ocr_enable=ocr_enable)
    T["finalize_middle_json"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    markdown = union_make(middle_json["pdf_info"], MakeMode.MM_MD, "images")
    T["union_make (markdown)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    md_dir = output_dir / stem / "auto"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_file = md_dir / f"{stem}.md"
    md_file.write_text(markdown, encoding="utf-8")
    T["write .md file"] = time.perf_counter() - t0

    # ── Top-level summary ────────────────────────────────────────────────────
    relevant = {k: v for k, v in T.items() if k != "model warm (ignored)"}
    total    = sum(relevant.values())
    print("╔" + "═"*55 + "╗")
    print("║  TOP-LEVEL PIPELINE TIMING" + " "*28 + "║")
    print("╠" + "═"*55 + "╣")
    for name, t in relevant.items():
        pct = t / total * 100
        bar = "█" * int(pct / 3)
        print(f"║  {name:<32} {t:6.2f}s  {pct:4.1f}%  ║")
    print("╠" + "═"*55 + "╣")
    print(f"║  {'TOTAL (excl. model warm)':<32} {total:6.2f}s        ║")
    print("╚" + "═"*55 + "╝")
    print(f"\n[OK] → {md_file}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 time_pipeline.py <pdf> [output_dir]")
        sys.exit(1)
    pdf   = args[0]
    outd  = args[1] if len(args) > 1 else str(Path(pdf).parent)
    run(pdf, outd)
