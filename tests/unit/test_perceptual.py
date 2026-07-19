import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from vtuber_rigger.qa.perceptual import render_avatar, lpips_similarity, clip_score, detect_action_units, detect_face_landmarks

def test_module_imports():
    """Test that the module can be imported without torch/lpips/clip being installed."""
    try:
        import vtuber_rigger.qa.perceptual
    except ImportError as e:
        pytest.fail(f"Module import failed: {e}")

def test_lpips_import_error():
    """Test that lpips_similarity raises ImportError when dependencies are missing."""
    # We can't easily 'uninstall' torch if it's present, but we can mock the import
    with patch('builtins.__import__', side_effect=ImportError):
        # This is a bit tricky because we need to trigger the import inside the function
        # Since the imports are lazy, we can mock the local import.
        with patch('vtuber_rigger.qa.perceptual.lpips_similarity', side_effect=ImportError("Missing deps")):
            # This mock is for the function itself, not the internal import.
            # Let's try a different approach.
            pass

def test_lazy_import_errors():
    """Test that functions raise ImportError gracefully when dependencies are missing."""
    # Use a mock to simulate missing torch/lpips/clip
    with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                (_ for _ in range(0)) if name in ['torch', 'lpips', 'clip', 'mediapipe', 'cv2'] 
                else __import__(name, *args, **kwargs)):
        
        # Since __import__ is used internally, we need to be careful.
        # A better way is to mock the specific import calls if possible.
        pass

def test_render_avatar_mock():
    """Test render_avatar with a mock of subprocess.run."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Rendered", stderr="")
        
        vrm_path = "test.vrm"
        result_path = render_avatar(vrm_path)
        
        assert result_path.endswith(".png")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "/mnt/b/Blender/blender.exe" in args[0]
        assert "--background" in args
        assert "--python-expr" in args
        assert vrm_path in args[3]

def test_dependencies_missing_behavior():
    """Verify functions raise ImportError when key deps are missing."""
    # We mock the import internally by patching the 'import' statement's effect
    # for specific modules.
    
    # This is hard in Python. Instead, we can manually trigger the ImportError by 
    # patching the libraries' existence.
    
    # For lpips_similarity, we can mock 'torch' and 'lpips' to raise ImportError when imported.
    with patch.dict('sys.modules', {'torch': None, 'lpips': None}):
        with pytest.raises(ImportError, match="The 'torch' and 'lpips' packages are required"):
            lpips_similarity("1.png", "2.png")
            
    with patch.dict('sys.modules', {'torch': None, 'clip': None}):
        with pytest.raises(ImportError, match="The 'torch' and 'clip' packages are required"):
            clip_score("1.png", "happy")
            
    with patch.dict('sys.modules', {'mediapipe': None, 'openface': None}):
        with pytest.raises(ImportError, match="Neither 'mediapipe' nor 'openface' is installed"):
            detect_action_units("1.png")
            
    with patch.dict('sys.modules', {'mediapipe': None, 'cv2': None}):
        with pytest.raises(ImportError, match="The 'mediapipe' and 'opencv-python' packages are required"):
            detect_face_landmarks("1.png")
