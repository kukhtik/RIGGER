import numpy as np
import cv2
import pytest
from PIL import Image
from vtuber_rigger.ml.layer_decomp import decompose_layers, Layer, LAYER_NAME_SET

def create_synthetic_image(path, size=(256, 256)):
    # Create a random RGB image
    img = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    # Add an alpha channel
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    cv2.imwrite(path, rgba)
    return path

@pytest.fixture
def test_image(tmp_path):
    img_path = str(tmp_path / "test_image.png")
    create_synthetic_image(img_path)
    return img_path

def test_layer_decomposition_invariants(test_image):
    layers = decompose_layers(test_image)
    
    # B1-1: 2 <= L <= 10
    assert 2 <= len(layers) <= 10
    
    h, w = layers[0].mask.shape
    
    # B1-2: Binary masks {0, 1}
    for layer in layers:
        assert np.all(np.isin(layer.mask, [0, 1]))
        assert layer.mask.dtype == np.uint8
    
    # B1-3: Coverage (sum masks == 1 everywhere)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for layer in layers:
        combined_mask += layer.mask
    
    # For a synthetic image we created with a full alpha channel, 
    # the coverage should be 1 everywhere.
    assert np.all(combined_mask == 1)
    
    # B1-4: Disjointness (masks are mutually exclusive)
    # Since we summed them and checked they equal 1, disjointness is implicit 
    # if we assume masks are binary {0,1}. 
    # But we can explicitly check pairs.
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            assert np.sum(layers[i].mask * layers[j].mask) == 0
            
    # B1-5: No empty layers
    for layer in layers:
        assert np.sum(layer.mask) > 0
        
    # B1-6: Valid names from LAYER_NAME_SET
    for layer in layers:
        assert layer.name in LAYER_NAME_SET
        
    # B1-7: Correct image dimensions HxWx4
    for layer in layers:
        assert layer.image.shape == (h, w, 4)
        
    # B1-8: Image alpha channel matches binary mask
    for layer in layers:
        alpha = layer.image[:, :, 3]
        # mask is {0,1}, alpha is {0,255}
        assert np.all(alpha == layer.mask * 255)

def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        decompose_layers("non_existent_file.png")

def test_invalid_path_fixed():
    with pytest.raises(FileNotFoundError):
        decompose_layers("non_existent_file.png")
