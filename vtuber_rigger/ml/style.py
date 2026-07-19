from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import cv2

from vtuber_rigger.interfaces import MeshData

@dataclass
class StyleResult:
    """Analysis of the visual style and material properties of the character."""
    palette: Dict[str, List[float]]  # region -> [r, g, b] in [0, 1]
    material_type: str              # 'toon', 'unlit', or 'pbr'
    textures: Dict[str, np.ndarray]  # region -> image (H, W, 3) uint8

def _extract_dominant_color(image: np.ndarray, k: int = 1) -> List[float]:
    """Extract the dominant color using k-means clustering."""
    # Image is BGR from OpenCV
    data = image.reshape((-1, 3)).astype(np.float32)
    
    # Simple k-means via OpenCV
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Return the most frequent center (which is the only center if k=1)
    dominant_color = centers[0]
    
    # Convert BGR [0, 255] to RGB [0, 1]
    rgb = dominant_color[::-1] / 255.0
    return rgb.tolist()

def _classify_material(image: np.ndarray) -> str:
    """Heuristic to classify material type based on color distribution/gradients."""
    # Convert to grayscale for contrast analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Toon materials typically have sharp boundaries (few values in mid-range of gradients)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian < 100: # Low variance in gradients = flat/toon
        return "toon"
    elif laplacian < 500:
        return "unlit"
    else:
        return "pbr"

def analyze_style(reference_images: List[str], mesh: Optional[MeshData] = None) -> StyleResult:
    """
    Analyze visual style and material properties from reference images.
    
    Args:
        reference_images: List of paths to image files.
        mesh: Optional mesh data for region mapping.
        
    Returns:
        StyleResult containing palette, material type, and basic textures.
    """
    # Lazy import for semantic classification if torch/CLIP is available
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    # Default palette
    regions = ["skin", "hair", "clothing"]
    palette = {region: [1.0, 1.0, 1.0] for region in regions}
    
    # We need at least one image to analyze
    if not reference_images:
        return StyleResult(palette=palette, material_type="toon", textures={})

    # Use the first image as the primary reference
    img_path = reference_images[0]
    image = cv2.imread(img_path)
    if image is None:
        return StyleResult(palette=palette, material_type="toon", textures={})

    # Simplified Region Extraction
    h, w, _ = image.shape
    
    # Crop approximations for demonstration of the logic
    crops = {
        "skin": image[int(h*0.2):int(h*0.4), int(w*0.3):int(w*0.7)],
        "hair": image[0:int(h*0.3), 0:w],
        "clothing": image[int(h*0.6):h, 0:w]
    }

    for region, crop in crops.items():
        if crop.size > 0:
            palette[region] = _extract_dominant_color(crop)

    # Material classification
    material_type = _classify_material(image)
    
    # Generate solid color texture maps (e.g., 256x256)
    textures = {}
    for region, color in palette.items():
        # Convert [0,1] RGB back to [0,255] BGR for OpenCV
        bgr_color = (np.array(color)[::-1] * 255).astype(np.uint8)
        textures[region] = np.full((256, 256, 3), bgr_color, dtype=np.uint8)

    return StyleResult(palette=palette, material_type=material_type, textures=textures)
