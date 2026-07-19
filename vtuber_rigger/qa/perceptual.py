import os
import subprocess
import tempfile
import numpy as np

def render_avatar(vrm_path: str, pose: str = 'front', expression: str = 'neutral') -> str:
    """
    Renders the avatar from a VRM file using Blender CLI.
    
    Args:
        vrm_path: Path to the .vrm file.
        pose: Pose to use for rendering (default 'front').
        expression: Expression to apply (default 'neutral').
        
    Returns:
        Path to the rendered PNG file.
    """
    blender_bin = "/mnt/b/Blender/blender.exe"
    
    # Use a permanent-ish temp file because Blender might need to write to it
    # and we want to return the path.
    fd, out_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    
    # This is a simplified python expression for Blender. 
    # In a real scenario, this would load the VRM, set the pose/expression, 
    # set up the camera and light, and render.
    python_expr = (
        f"import bpy; "
        f"import os; "
        f"bpy.ops.import_scene.vrm(filepath='{vrm_path}'); "
        f"# Set pose to {pose} and expression to {expression} (Mock logic); "
        f"bpy.context.scene.render.filepath='{out_path}'; "
        f"bpy.ops.render.render(write_still=True)"
    )
    
    try:
        subprocess.run(
            [blender_bin, "--background", "--python-expr", python_expr],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Blender render failed: {e.stderr}") from e
        
    return out_path

def lpips_similarity(img1_path: str, img2_path: str, model=None) -> float:
    """
    Computes LPIPS similarity between two images.
    Lower value means more similar.
    """
    try:
        import torch
        import lpips
    except ImportError:
        raise ImportError("The 'torch' and 'lpips' packages are required for lpips_similarity. Please install them via pip.")

    if model is None:
        model = lpips.LPIPS(net='alex')

    # Load and preprocess images
    from PIL import Image
    from torchvision.transforms import ToTensor, Normalize

    def load_img(path):
        img = Image.open(path).convert('RGB')
        img = img.resize((256, 256))
        t = ToTensor()(img)
        t = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(t)
        return t.unsqueeze(0)

    img1 = load_img(img1_path)
    img2 = load_img(img2_path)

    with torch.no_grad():
        dist = model(img1, img2)
    
    return dist.item()

def clip_score(img_path: str, text_label: str, model=None, preprocess=None) -> float:
    """
    Computes CLIP score between an image and a text label.
    Higher value means more semantically correct.
    """
    try:
        import torch
        import clip
    except ImportError:
        raise ImportError("The 'torch' and 'clip' packages are required for clip_score. Please install them via pip.")

    if model is None:
        model, preprocess = clip.load("ViT-B/32", device="cpu")

    from PIL import Image
    image = preprocess(Image.open(img_path)).unsqueeze(0)
    text = clip.tokenize([text_label])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
    
    return similarity

def detect_action_units(img_path: str) -> set[str]:
    """
    Detects facial action units from an image.
    """
    try:
        import mediapipe as mp
    except ImportError:
        try:
            import openface
        except ImportError:
            raise ImportError("Neither 'mediapipe' nor 'openface' is installed. Please install one of them for action unit detection.")

    # Mediapipe doesn't directly provide AUs (Action Units) like OpenFace.
    # Placeholder: return empty set for now.
    return set()

def detect_face_landmarks(img_path: str) -> np.ndarray | None:
    """
    Detects face landmarks using mediapipe.
    """
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        raise ImportError("The 'mediapipe' and 'opencv-python' packages are required for detect_face_landmarks.")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        image = cv2.imread(img_path)
        if image is None:
            return None
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        return coords
