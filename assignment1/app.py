from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        async with aiofiles.open(os.path.join(UPLOAD_DIRS, f"{image.filename}"), "wb") as file:
            await file.write(contents)

        result = run_custom_font_detector(os.path.join(UPLOAD_DIRS, image.filename))
        logger.info(result)
    except Exception as e:
        logger.error(f"Server error observed: {e}")
        return HTTPException(status_code=500, detail=f"Server error: {e}")
    finally:
        await image.close()
    image_path = os.path.join("assets", "image_bbox_final.png")
    return FileResponse(path=image_path)

@app.get("/home")
async def home():
    content = '''
  <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Upload and Display</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h2 {
            margin-top: 0;
        }
        form {
            margin-bottom: 20px;
            text-align: center;
        }
        input[type="file"] {
            display: none;
        }
        .custom-file-upload {
            display: inline-block;
            cursor: pointer;
            padding: 10px 20px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            transition: background-color 0.3s ease;
        }
        .custom-file-upload:hover {
            background-color: #0056b3;
        }
        #output {
            text-align: center;
        }
        img {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 4px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Upload Image</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <label for="imageFile" class="custom-file-upload">Choose File</label>
            <input type="file" name="imageFile" id="imageFile" accept="image/*" onchange="previewImage(event)">
            <button type="submit">Upload</button>
        </form>
        <div id="output">
            <!-- Image output will be displayed here -->
        </div>
    </div>

    <script>
        function previewImage(event) {
            const input = event.target;
            const reader = new FileReader();

            reader.onload = function(){
                const output = document.getElementById('output');
                output.innerHTML = `<img src="${reader.result}" alt="Preview Image">`;
            };

            reader.readAsDataURL(input.files[0]);
        }

        document.getElementById('uploadForm').addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData();
            formData.append('image', document.getElementById('imageFile').files[0]);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.blob())
            .then(blob => {
                const imageUrl = URL.createObjectURL(blob);
                document.getElementById('output').innerHTML = `<img src="${imageUrl}" alt="Uploaded Image">`;
            })
            .catch(error => {
                console.error(error);
                alert('Error: ' + error.message);
            });
        });
    </script>
</body>
</html>


    '''
    return HTMLResponse(content=content)