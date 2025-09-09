import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import face_recognition
import numpy as np
from face_recognition import face_locations
from fastapi import UploadFile, FastAPI, HTTPException, Request, File, Form
from pydantic import BaseModel
from typing import List, Tuple
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

def evaluate_pic(img):

    gray = np.mean(img, axis=2)
    std_dev = np.std(gray)

    print(std_dev)

    if std_dev < 30:
        raise HTTPException(status_code=400, detail="Low contrast")
    elif std_dev > 50:
        raise HTTPException(status_code=400, detail="High contrast")

    #loop = asyncio.get_running_loop()
    face_locations = model_call(img)

    #face_locations = await loop.run_in_executor(executor, lambda: model_call(img))

    if len(face_locations) < 1:
        raise HTTPException(status_code=400, detail="No face found")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Too many faces found")

    h, w, c = img.shape
    x_left_bottom = face_locations[0][3]
    y_left_bottom = face_locations[0][2]
    x_right_top = face_locations[0][1]
    y_right_top = face_locations[0][0]
    #print(x_left_bottom, y_left_bottom, x_right_top, y_right_top)
    width = x_right_top - x_left_bottom
    height = y_left_bottom - y_right_top

    if (width * height) / (h * w) < 0.3:
        raise HTTPException(status_code=400, detail="Area of bbox too small")
    return img, (x_left_bottom, y_left_bottom, x_right_top, y_right_top)

# async def async_prepare_image(img: UploadFile):
#     loop = asyncio.get_running_loop()
#     return await loop.run_in_executor(None, lambda: prepare_image(img))

def prepare_image(img):
    if not (img.filename.lower().endswith(".png") or img.filename.lower().endswith(".jpg")): raise HTTPException(status_code=400,
                                                                       detail="Only PNG/JPG files are allowed")
    contents = img.file.read()
    image = Image.open(BytesIO(contents))
    image = image.convert("RGB")
    return np.array(image)

async def detect_faces(img):
    img_np = prepare_image(img)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: evaluate_pic(img_np))

class PredictionResponse(BaseModel):
    picture: str
    bbox: Tuple[int, int, int, int]

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
    #print(next(iter(results)))
    response_list = [
        PredictionResponse(
            picture=file.filename,
            bbox=bbox
        )
        for file, (img, bbox) in zip(files, results)  # распаковываем кортеж (img, bbox)
    ]

    return response_list
