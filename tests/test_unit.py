import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from pathlib import Path
from PIL import Image
import pytest
from fastapi import HTTPException, UploadFile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_call():

    from app.main import model_call
    import app.main
    import types

    app.main.app.state = types.SimpleNamespace()
    app.main.app.state.model_lock = threading.Lock()
    app.main.app.state.executor = ThreadPoolExecutor(max_workers=1)

    current_dir = Path(__file__).parent
    img_path = current_dir.parent / "tests" / "resources" / "test.jpg"
    img = Image.open(img_path).convert("RGB")
    np_img = np.array(img)
    result = model_call(np_img)
    assert isinstance(result, list)
    assert all(len(box) == 4 for box in result)
    assert len(result) >= 1

def test_evaluate():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    from app.main import evaluate_pic
    import app.main
    import types
    from config.config import Settings
    app.main.app.state = types.SimpleNamespace()
    app.main.app.state.settings = Settings()
    with pytest.raises(HTTPException) as exc_info:
        evaluate_pic(img)
    assert exc_info.value.status_code == 400
    assert "Low contrast" in exc_info.value.detail

def test_prepare_image():
    from app.main import prepare_image
    img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img_array)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    upload_file = UploadFile(filename="test.png", file=buf)
    result = prepare_image(upload_file)

    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)

@pytest.mark.asyncio
async def test_process_image():

    from app.main import process_image
    import types
    from config.config import Settings
    import app.main

    img_array = np.full((100, 100, 3), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img_array)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    upload_file = UploadFile(filename="test.png", file=buf)

    app.main.app.state = types.SimpleNamespace()
    app.main.app.state.settings = Settings()

    with pytest.raises(HTTPException) as exc_info:
        await process_image(upload_file, "prompt")

    assert exc_info.value.status_code == 400
    assert "Wrong image size" in exc_info.value.detail