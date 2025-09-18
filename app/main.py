import asyncio
import base64
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from config.config import Settings
import face_recognition
import numpy as np
from fastapi import UploadFile, FastAPI, HTTPException, Request, File, Form
from pydantic import BaseModel
from typing import List, Tuple
from fastapi.responses import JSONResponse
from PIL import Image
from contextlib import asynccontextmanager
from huggingface_hub import InferenceClient

@asynccontextmanager
async def lifespan(app: FastAPI):

    executor = ThreadPoolExecutor(max_workers=1)

    model_lock = threading.Lock()

    app.state.executor = executor

    app.state.model_lock = model_lock

    app.state.settings = Settings()

    app.state.client = InferenceClient(

        provider = app.state.settings.provider,

        api_key = app.state.settings.api_key,
    )

    yield

    executor.shutdown()

def model_call(img : np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Run face detection model on an image with thread-safety.

    Args:
        img (np.ndarray): RGB image as numpy array.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes (top, right, bottom, left).

    Raises:
        HTTPException: If inference fails.
    """
    with app.state.model_lock:
        try:
            return face_recognition.face_locations(img)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

def evaluate_pic(img : np.ndarray)-> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Evaluate image quality and run face detection.

    Args:
        img (np.ndarray): RGB image as numpy array.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]: Image and one bounding box (x_left, y_bottom, x_right, y_top).

    Raises:
        HTTPException: If image contrast is invalid, no faces found, too many faces, or face bbox too small.
    """

    gray = np.mean(img, axis=2)
    std_dev = np.std(gray)

    if std_dev < app.state.settings.lower_threshold:
        raise HTTPException(status_code=400, detail="Low contrast")
    elif std_dev > app.state.settings.upper_threshold:
        raise HTTPException(status_code=400, detail="High contrast")

    face_locations = model_call(img)

    if len(face_locations) < 1:
        raise HTTPException(status_code=400, detail="No face found")
    if len(face_locations) > 1:
        raise HTTPException(status_code=400, detail="Too many faces found")

    picture_height, picture_weights, channels = img.shape
    x_left_bottom = face_locations[0][3]
    y_left_bottom = face_locations[0][2]
    x_right_top = face_locations[0][1]
    y_right_top = face_locations[0][0]
    width = x_right_top - x_left_bottom
    height = y_left_bottom - y_right_top

    if (width * height) / (picture_height * picture_weights) < app.state.settings.area_factor:
        raise HTTPException(status_code=400, detail="Area of bbox too small")
    return img, (x_left_bottom, y_left_bottom, x_right_top, y_right_top)

def prepare_image(img : UploadFile) -> np.ndarray:
    """
    Validate and convert uploaded file into numpy RGB image.

    Args:
        img (UploadFile): Uploaded file.

    Returns:
        np.ndarray: Image as numpy array.

    Raises:
        HTTPException: If file format is not PNG/JPG.
    """

    if not (img.filename.lower().endswith(".png") or img.filename.lower().endswith(".jpg")): raise HTTPException(status_code=400,
                                                                       detail="Only PNG/JPG files are allowed")
    contents = img.file.read()
    image = Image.open(BytesIO(contents))
    image = image.convert("RGB")
    return np.array(image)

async def detect_faces(img : UploadFile) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Run face detection on uploaded image in background thread.

    Args:
        img (UploadFile): Uploaded file.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]: Image and bounding box.
    """
    img_np = prepare_image(img)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(app.state.executor, lambda: evaluate_pic(img_np))


async def process_image(file: UploadFile, prompt: str) -> str:
    """
    Processes an uploaded image file: validates its format and resolution,
    sends it to the InferenceClient for image-to-image generation,
    and returns the generated image as a base64 string.

    Args:
        file (UploadFile): The uploaded image file (.png or .jpg)
        prompt (str): Text prompt for image generation

    Raises:
        HTTPException: If the file format is not supported
        HTTPException: If the prompt length exceeds the maximum allowed
        HTTPException: If the prompt contains no letters
        HTTPException: If the image resolution is smaller than the minimum required

    Returns:
        str: Generated image encoded as a base64 string
    """
    if not (file.filename.lower().endswith(".png") or file.filename.lower().endswith(".jpg")): raise HTTPException(
        status_code=400,
        detail="Only PNG/JPG files are allowed")

    if len(prompt) > app.state.settings.prompt_max_length: raise HTTPException(status_code=400,

                                                                               detail="Prompt is too long")
    if not re.search(r"[a-zA-Z]", prompt):
        raise HTTPException(
            status_code=400,
            detail="Prompt must contain at least one letter"
        )

    contents = await file.read()

    with Image.open(BytesIO(contents)) as img:
        width, height = img.size

    if (width < app.state.settings.min_width) or (height < app.state.settings.min_height):
        raise HTTPException(status_code=400, detail="Wrong image size")

    loop = asyncio.get_running_loop()
    image: Image.Image = await loop.run_in_executor(
        None,
        lambda: app.state.client.image_to_image(contents, prompt=prompt, model="Qwen/Qwen-Image-Edit")
    )

    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded

class PredictionResponse(BaseModel):
    """
    Response model for face detection.

    Attributes:
        picture (str): Base64-encoded image.
        bbox (Tuple[int, int, int, int]): Bounding box (x_left, y_bottom, x_right, y_top).
    """
    picture: bytes
    bbox: Tuple[int, int, int, int]

class GenerationResponse(BaseModel):
    """
    Response model for face detection.

    Attributes:
        picture (str): Base64-encoded image.
        bbox (Tuple[int, int, int, int]): Bounding box (x_left, y_bottom, x_right, y_top).
    """
    picture: bytes

app = FastAPI(lifespan=lifespan)

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
) -> List[PredictionResponse]:
    """
    Upload multiple files, run face detection, and return results.

    Args:
        files (List[UploadFile]): List of uploaded image files.
        action (str, optional): Mock parameter for additional actions. Defaults to "default_action".

    Returns:
        List[PredictionResponse]: List of responses with base64-encoded images and bounding boxes.
    """
    tasks = [detect_faces(file) for file in files]
    results = await asyncio.gather(*tasks)
    response_list = []
    for file, (img, bbox) in zip(files, results):
        file.file.seek(0)
        file_bytes = await file.read()
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        response_list.append(PredictionResponse(picture=encoded, bbox=bbox))
    return response_list

@app.post("/uploadfiles_")
async def create_upload_files(
        files: List[UploadFile] = File(...),
        prompt: str = Form(...),
) -> List[GenerationResponse]:
    """
    Uploads multiple image files and processes them asynchronously
    using the provided prompt. Returns a list of generated images in base64 format.

    Args:
        files (List[UploadFile]): List of uploaded image files (.png or .jpg)
        prompt (str): Text prompt for generating images

    Returns:
        List[GenerationResponse]: List of response objects containing the base64-encoded images
    """
    tasks = [process_image(file, prompt) for file in files]
    results = await asyncio.gather(*tasks)
    response_list = []
    for res in results:
        response_list.append(GenerationResponse(picture=res))
    return response_list