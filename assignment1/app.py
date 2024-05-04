from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.exceptions import HTTPException
from wrapper import run_custom_font_detector
import logging
import os
from uuid import uuid4
import shutil
import aiofiles

logger = logging.getLogger(__name__)

app = FastAPI()
UPLOAD_DIRS = "upload_dirs"
if not os.path.exists(UPLOAD_DIRS):
    os.makedirs(UPLOAD_DIRS, exist_ok=True)

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        async with aiofiles.open(os.path.join(UPLOAD_DIRS, f"{image.filename}"), "wb") as file:
            await file.write(contents)

        # with open(os.path.join(UPLOAD_DIRS, f"{image.filename}"), "wb") as file:
        #     shutil.copyfileobj(image.file, file)

        result = run_custom_font_detector(os.path.join(UPLOAD_DIRS, image.filename))
        logger.info(result)
    except Exception as e:
        logger.error(f"Server error observed: {e}")
        return HTTPException(status_code=500, detail=f"Server error: {e}")
    finally:
        await image.close()
    
    return FileResponse(path="assets/image_bbox_final.png", media_type="image/png")
@app.get("/home")
async def home():
    content = '''
    <body>
    <form action='/upload' enctype='multipart/form-data' method='post'>
    <input name='image' type='file'>
    <input type='submit'>
    </form>
    </body>
    '''
    return HTMLResponse(content=content)