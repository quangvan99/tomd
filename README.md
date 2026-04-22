# tomd

PDF → Markdown. Chạy hoàn toàn trong Docker (GPU).

## Setup

```bash
./start.sh        # build image (lần đầu) + chạy server uvicorn tại :9000
./log.sh          # xem log
./stop.sh         # tắt container
```

## Cách dùng

### 1. CLI

```bash
./run.sh <pdf_path> [output_dir]
```

- **Input**: đường dẫn 1 file `.pdf` hoặc 1 thư mục chứa nhiều `.pdf`.
- **Output** (mặc định cùng thư mục input, hoặc `output_dir` nếu truyền):
  ```
  <output_dir>/<stem>/auto/<stem>.md
  <output_dir>/<stem>/auto/images/...
  ```

Ví dụ:

```bash
./run.sh raw/PMC6375377.pdf
./run.sh raw/ output/
```

### 2. HTTP API

Server tự chạy sau `./start.sh` tại `http://localhost:9000`.

```bash
curl -X POST http://localhost:9000/convert \
  -H "Content-Type: application/json" \
  -d '{"url": "raw/PMC6375377.pdf"}'
```

- **Input (JSON body)**:
  - `url` (required): local path hoặc URL tới file PDF.
  - `lang` (optional, default `"en"`): ngôn ngữ OCR.
  - `parse_method` (optional, default `"auto"`): `"auto"` | `"ocr"` | `"txt"`.
- **Output**: JSON chứa markdown + đường dẫn file `.md` đã ghi ra đĩa (trong thư mục output của server).
