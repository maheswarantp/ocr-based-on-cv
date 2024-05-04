from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from wrapper import run_custom_font_detector
import logging
import os
from uuid import uuid4
import shutil

logger = logging.getLogger(__name__)

app = FastAPI()
UPLOAD_DIRS = "upload_dirs"
if not os.path.exists(UPLOAD_DIRS):
    os.makedirs(UPLOAD_DIRS, exist_ok=True)

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        with open(os.path.join(UPLOAD_DIRS, f"{image.filename}"), "wb") as file:
            shutil.copyfileobj(image.file, file)

        result = run_custom_font_detector(os.path.join(UPLOAD_DIRS, image.filename))
        logger.info(result)
        return JSONResponse({"response":f"success, file uploaded as {image.filename}"})
    except Exception as e:
        logger.error(f"Server error observed: {e}")
        return HTTPException(status_code=500, detail=f"Server error: {e}")


@app.post("/home")
async def home():
    return JSONResponse({
        "response":"success"
    })
