import pytest
import numpy as np
import cv2
import os
from vtuber_rigger.ml.style import analyze_style, StyleResult

@pytest.fixture
def dummy_image(tmp_path):
    # Create a simple 256x256 image with some colored blocks
    # BGR format for OpenCV
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Hair: Top (Blue)
    img[0:70, :] = [255, 0, 0] 
    # Skin: Middle (Beige-ish)
    img[70:120, 80:170] = [200, 200, 150]
    # Clothing: Bottom (Green)
    img[150:, :] = [0, 255, 0]
    
    img_path = str(tmp_path / "test_ref.png")
    cv2.imwrite(img_path, img)
    return img_path

def test_analyze_style_basic(dummy_image):
    res = analyze_style([dummy_image])
    
    assert isinstance(res, StyleResult)
    assert "skin" in res.palette
    assert "hair" in res.palette
    assert "clothing" in res.palette
    
    # Check color range [0, 1]
    for color in res.palette.values():
        for channel in color:
            assert 0.0 <= channel <= 1.0
            
    # Check material type validity
    assert res.material_type in ["toon", "unlit", "pbr"]
    
    # Check textures
    assert "skin" in res.textures
    assert res.textures["skin"].shape == (256, 256, 3)
    assert res.textures["skin"].dtype == np.uint8

def test_analyze_style_empty_images():
    res = analyze_style([])
    assert res.material_type == "toon"
    assert res.palette["skin"] == [1.0, 1.0, 1.0]

def test_analyze_style_invalid_image(tmp_path):
    bad_path = str(tmp_path / "bad.png")
    with open(bad_path, "w") as f:
        f.write("not an image")
    
    res = analyze_style([bad_path])
    assert res.material_type == "toon"
    assert res.textures == {}

def test_analyze_style_mesh_ignored(dummy_image):
    # Testing that it doesn't crash when mesh is provided
    from vtuber_rigger.interfaces import MeshData
    mesh = MeshData(
        vertices=np.zeros((10, 3), dtype=np.float32),
        faces=np.zeros((10, 3), dtype=np.int32)
    )
    res = analyze_style([dummy_image], mesh=mesh)
    assert isinstance(res, StyleResult)
