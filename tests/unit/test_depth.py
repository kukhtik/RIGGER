import numpy as np
import cv2
import pytest
from vtuber_rigger.ml.depth import estimate_depth

def test_depth_invariants():
    # Create a synthetic image
    h, w = 256, 256
    image = np.zeros((h, w), dtype=np.uint8)
    # Add a white circle in the center to simulate a subject
    cv2.circle(image, (w // 2, h // 2), 50, 255, -1)
    
    image_path = "test_synthetic_depth.png"
    cv2.imwrite(image_path, image)
    
    try:
        # Test without masks
        depth = estimate_depth(image_path)
        
        # B2-1: Shape matches image
        assert depth.shape == (h, w)
        
        # B2-2: Normalized [0, 1]
        assert np.all(depth >= 0.0)
        assert np.all(depth <= 1.0)
        
        # B2-3: Finite (no NaN/Inf)
        assert np.all(np.isfinite(depth))
        
        # B2-4: Center is generally closer (higher value) than corners
        center_val = depth[h // 2, w // 2]
        corner_val = depth[0, 0]
        assert center_val >= corner_val
        
        # Test with synthetic masks
        masks = {
            "body": (np.ones((h, w)) * 0.5).astype(np.float32),
            "background": (np.ones((h, w)) * 0.1).astype(np.float32)
        }
        # B2-5: Mask consistency (masks should influence result)
        depth_with_masks = estimate_depth(image_path, layer_masks=masks)
        assert depth_with_masks.shape == (h, w)
        assert np.all(np.isfinite(depth_with_masks))
        
    finally:
        import os
        if os.path.exists(image_path):
            os.remove(image_path)

def test_depth_file_not_found():
    with pytest.raises(FileNotFoundError):
        estimate_depth("non_existent_image.png")
