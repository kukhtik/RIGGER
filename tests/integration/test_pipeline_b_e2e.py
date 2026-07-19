import os
import numpy as np
from PIL import Image
import pytest
from vtuber_rigger.pipeline.b_pipeline import run_pipeline_b
from vtuber_rigger.interfaces import VRMMeta

@pytest.mark.skip(reason="Synthetic random image doesn't produce enough geometry for reconstruction. Needs real VTuber screenshot.")
def test_pipeline_b_e2e_synthetic():
    """Test Path B with synthetic screenshot data."""
    # Create a dummy image
    img_path = "synthetic_screenshot.png"
    img = Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
    img.save(img_path)
    
    try:
        screenshots = {"front": img_path}
        output_path = "output_test_b.vrm"
        meta = VRMMeta(name="test_model")
        
        result = run_pipeline_b(screenshots, output_path, meta)
        
        # Since the underlying ML/Rigging modules are likely mocks or 
        # return standard objects in this environment, we check for successful flow.
        assert result.exit_code == 0
        assert "decompose_layers" in result.stage_times
        assert "export_vrm" in result.stage_times
        assert any("reduced quality" in w for w in result.warnings)
        
    finally:
        # Cleanup
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists("output_test_b.vrm"):
            os.remove("output_test_b.vrm")

def test_pipeline_b_no_screenshots():
    """Test Path B with empty screenshots dictionary."""
    result = run_pipeline_b({}, "fail.vrm")
    assert result.exit_code == 1
    assert "No screenshots provided" in result.errors[0]
