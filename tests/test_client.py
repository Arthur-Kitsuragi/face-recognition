from fastapi.testclient import TestClient
import io
from fastapi import UploadFile
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.main
client = TestClient(app.main.app)

def test_uploadfiles_endpoint(monkeypatch):

    async def dummy_detect_faces(file):
        import numpy as np
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_bbox = (0, 9, 9, 0)
        return dummy_img, dummy_bbox

    monkeypatch.setattr("app.main.detect_faces", dummy_detect_faces)

    import numpy as np
    from PIL import Image

    img_array = np.zeros((10, 10, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/uploadfiles",
        files={"files": ("test.png", buf, "image/png")}
    )

    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert "picture" in json_data[0]
    assert "bbox" in json_data[0]
    assert json_data[0]["bbox"] == [0, 9, 9, 0]  # совпадает с моком