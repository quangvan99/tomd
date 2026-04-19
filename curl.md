uvicorn server:app --host 0.0.0.0 --port 9000


curl -X POST http://localhost:9000/convert \
  -H "Content-Type: application/json" \
  -d '{"url": "raw/PMC6375377.pdf"}' \
  -w "\nTime Total: %{time_total}s\n"


curl -X POST http://localhost:9000/convert \
  -H "Content-Type: application/json" \
  -d '{"url": "raw/2501.00852v1.pdf"}' \
  -w "\nTime Total: %{time_total}s\n"