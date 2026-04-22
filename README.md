# tomd

PDF → Markdown. Chạy hoàn toàn trong Docker (GPU). Toàn bộ thư mục hiện tại được bind-mount vào `/app` trong container.

## Setup

```bash
./start.sh        # build image (lần đầu) + chạy server uvicorn tại :9000
./log.sh          # xem log
./stop.sh         # tắt container (tắt cả uvicorn)
```

## Cách dùng

### 1. CLI

```bash
./run.sh <pdf_path> [output_dir]
```

- **Input**: 1 file `.pdf` hoặc 1 thư mục chứa nhiều `.pdf`. Path là relative tới thư mục repo (được mount vào `/app`).
- **Output** (mặc định cùng thư mục input, hoặc `output_dir` nếu truyền):
  ```
  <output_dir>/<stem>/auto/<stem>.md
  <output_dir>/<stem>/auto/images/...
  ```

Ví dụ:

```bash
./run.sh raw/PMC6375377.pdf
./run.sh raw/ out/
```

### 2. HTTP API

Server tự chạy sau `./start.sh` tại `http://localhost:9000`.

**Health check:**

```bash
curl http://localhost:9000/health
# {"status":"ok"}
```

**Convert:**

```bash
curl -X POST http://localhost:9000/convert \
  -H "Content-Type: application/json" \
  -d '{"url": "raw/PMC6375377.pdf"}'
```

- **Input (JSON body)**:
  - `url` (required): local path (relative tới repo) hoặc URL tới file PDF.
  - `lang` (optional, default `"en"`): ngôn ngữ OCR.
  - `parse_method` (optional, default `"auto"`): `"auto"` (tự phát hiện), `"ocr"` (ép OCR), hoặc giá trị khác (text mode).
- **Output**:
  - File `.md` + thư mục `images/` được ghi vào `raw/<stem>/auto/` (hardcoded).
  - Response JSON: `{"markdown_file": "/app/raw/<stem>/auto/<stem>.md", "pages": <n>}`.

## Lưu ý

Container chạy user `root`, nên file output thuộc `root:root` trên host. Xóa/sửa cần `sudo` hoặc `docker compose exec tomd rm ...`.
