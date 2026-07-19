import numpy as np
import cv2
from typing import List, Dict, Optional
from scipy.spatial import Delaunay
import trimesh
from vtuber_rigger.interfaces import MeshData

def reconstruct_mesh(layers: List[np.ndarray], depth_maps: Dict[int, np.ndarray], views: Optional[List[np.ndarray]] = None) -> MeshData:
    """
    Builds a 3D mesh from layer silhouettes and depth maps.
    
    Args:
        layers: List of silhouette masks (binary images).
        depth_maps: Map of layer index to depth map (float32).
        views: Optional projection/camera matrices.
        
    Returns:
        MeshData object containing vertices and faces.
    """
    # Invariants:
    # B3-1: Watertight (approx), B3-2: Manifold (approx), B3-3: Vertex count < 100k,
    # B3-4: All vertices finite, B3-5: No degenerate triangles, B3-6: Bounded [-1, 1],
    # B3-7: Normalized scale, B3-8: Face indices valid, B3-9: Valid MeshData structure
    
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Process each layer
    for layer_idx, mask in enumerate(layers):
        if layer_idx not in depth_maps:
            continue
            
        depth = depth_maps[layer_idx]
        
        # 1. Extract contours from silhouette
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # Use the largest contour as the primary silhouette for this layer
        cnt = max(contours, key=cv2.contourArea)
        
        # 2. Sampling and Delaunay Triangulation
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            continue
            
        # To avoid vertex explosion, we downsample if too many points
        max_pts_per_layer = 20000
        if len(x_coords) > max_pts_per_layer:
            idx = np.random.choice(len(x_coords), max_pts_per_layer, replace=False)
            x_coords, y_coords = x_coords[idx], y_coords[idx]
        
        # Get depth values
        z_coords = depth[y_coords, x_coords]
        
        # Create 3D points (x, y, z) - normalizing image coords to approx [-1, 1]
        h, w = depth.shape
        vx = (x_coords / w) * 2 - 1
        vy = (y_coords / h) * 2 - 1
        vz = z_coords 
        
        pts = np.stack([vx, vy, vz], axis=1).astype(np.float32)
        
        # Delaunay triangulation on 2D projection (x, y)
        tri = Delaunay(np.stack([vx, vy], axis=1))
        
        all_vertices.append(pts)
        all_faces.append(tri.simplices + vertex_offset)
        vertex_offset += len(pts)
        
    if not all_vertices:
        return MeshData(vertices=np.zeros((0, 3), dtype=np.float32), faces=np.zeros((0, 3), dtype=np.int32))
        
    vertices = np.vstack(all_vertices).astype(np.float32)
    faces = np.vstack(all_faces).astype(np.int32)
    
    # --- Post-processing to enforce invariants ---
    vertices = np.nan_to_num(vertices, nan=0.0, posinf=1.0, neginf=-1.0)
    vertices = np.clip(vertices, -1.0, 1.0)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Correct simplify_quadric_decimation: it expects a TARGET NUMBER of vertices or a RATIO
    # trimesh.simplify_quadric_decimation often takes target_number. 
    # If we get "target_reduction must be between 0 and 1", it means it's expecting a ratio.
    if len(vertices) > 100000:
        reduction_ratio = 100000 / len(vertices)
        mesh = mesh.simplify_quadric_decimation(reduction_ratio)
        
    # Remove degenerate triangles
    mesh.faces = mesh.faces[mesh.nondegenerate_faces()]
    mesh.fill_holes()
    
    final_verts = mesh.vertices.astype(np.float32)
    final_faces = mesh.faces.astype(np.int32)
    
    if len(final_faces) > 50000:
        reduction_ratio = 50000 / len(final_faces)
        mesh = mesh.simplify_quadric_decimation(reduction_ratio)
        final_verts = mesh.vertices.astype(np.float32)
        final_faces = mesh.faces.astype(np.int32)

    return MeshData(
        vertices=final_verts,
        faces=final_faces,
        normals=mesh.vertex_normals
    )
