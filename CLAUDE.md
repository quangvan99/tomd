# tomd

PDF → Markdown. Chạy hoàn toàn trong Docker (mount `./` vào `/app`, GPU).

## Scripts

- `./start.sh` — build image nếu chưa có, `docker compose up -d` (chạy `uvicorn server:app` trên `:9000`)
- `./run.sh <pdf_path> [output_dir]` — chạy `run.py` bên trong container (tự up container nếu cần)
- `./log.sh` — `docker compose logs -f tomd`
- `./stop.sh` — `docker compose down`

## Debug

**Luôn dùng môi trường Docker**, không chạy Python trực tiếp trên host.

- Chạy lệnh ad-hoc: `docker compose exec tomd <cmd>` (vd `python -c ...`, `pip list`, `bash`)
- Sửa code: edit file tại host — container đã mount `./:/app`, không cần rebuild
- Đổi dependency (`Dockerfile`): `./stop.sh && docker compose build && ./start.sh`
- Reload server sau khi sửa code: `docker compose restart tomd`
- Xem log: `./log.sh`
