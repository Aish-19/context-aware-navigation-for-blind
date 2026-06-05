import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import analyze_image


app = FastAPI(title="Path Guidance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/guide")
async def guide(image: UploadFile = File(...)):
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await image.read())
            temp_path = temp_file.name

        return analyze_image(temp_path)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not analyze image: {error}",
        ) from error
    finally:
        if "temp_path" in locals():
            Path(temp_path).unlink(missing_ok=True)
