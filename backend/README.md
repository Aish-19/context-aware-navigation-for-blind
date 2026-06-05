# Path Guidance Backend

FastAPI wrapper around the Qwen vision-language guidance model.

## Setup

From the project root:

```bash
.venv/bin/python -m pip install -r backend/requirements.txt
```

## Run

```bash
cd backend
../.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Test

```bash
curl http://127.0.0.1:8000/health
curl -F "image=@../frame.jpg" http://127.0.0.1:8000/guide
```

For an Android emulator, use `http://10.0.2.2:8000` as the backend URL.
For a physical phone, use your Mac's LAN IP address, for example `http://192.168.1.20:8000`.
