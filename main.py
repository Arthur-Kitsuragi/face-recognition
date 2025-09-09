import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import face_recognition
import numpy as np
from fastapi import UploadFile, FastAPI, HTTPException, Request, File, Form
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from PIL import Image

model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

def model_call(img) -> str:
    with model_lock:
        try:
            return face_recognition.face_locations(img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

async def evaluate_pic(img):

    gray = np.mean(img, axis=2)
    std_dev = np.std(gray)

    if std_dev < 30:
        raise HTTPException(status_code=400, detail="Low contrast")
    elif std_dev > 120:
        raise HTTPException(status_code=400, detail="High contrast")

    loop = asyncio.get_running_loop()
    face_locations = await loop.run_in_executor(executor, lambda: model_call(img))

    if len(face_locations) < 1:
        raise HTTPException(status_code=400, detail="No face found")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Too many faces found")

    h, w, c = img.shape
    x_left_bottom = face_locations[0][3]
    y_left_bottom = face_locations[0][2]
    x_right_top = face_locations[0][1]
    y_right_top = face_locations[0][0]
    width = x_right_top - x_left_bottom
    height = y_left_bottom - y_right_top

    if (width * height) / (h * w) < 0.3:
        raise HTTPException(status_code=400, detail="Area of bbox too small")
    return img, face_locations

async def async_prepare_image(img: UploadFile):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: prepare_image(img))

def prepare_image(img):
    if not (img.filename.lower().endswith(".png") or img.filename.lower().endswith(".jpg")): raise HTTPException(status_code=400,
                                                                       detail="Only PNG/JPG files are allowed")
    contents = img.file.read()
    image = Image.open(BytesIO(contents))
    image = image.convert("RGB")
    return np.array(image)

async def detect_faces(img):
    img_np = await async_prepare_image(img)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(app.state.executor, lambda: evaluate_pic(img_np))

class PredictionResponse(BaseModel):
    picture: str
    bbox: str

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Global HTTPException error handler.

    Args:
    request (Request): Request object.
    exc (HTTPException): FastAPI exception.

    Returns:
    JSONResponse: JSON with error description.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "url": str(request.url)
        }
    )

@app.post("/uploadfiles")
async def create_upload_files(
    files: List[UploadFile] = File(...),
    action: str = Form("default_action")  #mock
):
    tasks = [detect_faces(file) for file in files]
    results = await asyncio.gather(*tasks)

    response_list = [
        PredictionResponse(
            picture=file.filename,
            bbox={"x": bbox[0], "y": bbox[1], "x1": bbox[2], "y1": bbox[3]}
        )
        for file, (img, bbox) in zip(files, results)  # распаковываем кортеж (img, bbox)
    ]

    return response_list
