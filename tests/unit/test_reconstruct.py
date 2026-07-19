import numpy as np
import pytest
from vtuber_rigger.ml.reconstruct import reconstruct_mesh
from vtuber_rigger.interfaces import MeshData

def test_reconstruct_mesh_synthetic():
    # Create synthetic layers (binary masks)
    # 2 layers, 100x100 resolution
    h, w = 100, 100
    layer0 = np.zeros((h, w), dtype=np.uint8)
    layer0[20:80, 20:80] = 1 # A square in the center
    
    layer1 = np.zeros((h, w), dtype=np.uint8)
    layer1[30:70, 30:70] = 1 # A smaller square inside
    
    layers = [layer0, layer1]
    
    # Create synthetic depth maps
    # depth0: flat at z=0.5
    # depth1: flat at z=0.6
    depth0 = np.full((h, w), 0.5, dtype=np.float32)
    depth1 = np.full((h, w), 0.6, dtype=np.float32)
    depth_maps = {0: depth0, 1: depth1}
    
    mesh_data = reconstruct_mesh(layers, depth_maps)
    
    assert isinstance(mesh_data, MeshData)
    assert mesh_data.vertices.shape[1] == 3
    assert mesh_data.faces.shape[1] == 3
    
    # B3-3: Vertex count < 100k
    assert mesh_data.vertex_count < 100000
    
    # B3-4: Finite
    assert np.all(np.isfinite(mesh_data.vertices))
    
    # B3-6: Bounded [-1, 1]
    assert np.all(mesh_data.vertices >= -1.0)
    assert np.all(mesh_data.vertices <= 1.0)
    
    # B3-8: Valid indices
    max_idx = mesh_data.vertex_count - 1
    assert np.all(mesh_data.faces <= max_idx)
    assert np.all(mesh_data.faces >= 0)

def test_reconstruct_mesh_empty():
    layers = [np.zeros((100, 100), dtype=np.uint8)]
    depth_maps = {0: np.zeros((100, 100), dtype=np.float32)}
    
    mesh_data = reconstruct_mesh(layers, depth_maps)
    
    assert isinstance(mesh_data, MeshData)
    assert mesh_data.vertex_count == 0
    assert mesh_data.face_count == 0

def test_reconstruct_mesh_no_depth():
    layers = [np.ones((100, 100), dtype=np.uint8)]
    depth_maps = {} # Missing depth for layer 0
    
    mesh_data = reconstruct_mesh(layers, depth_maps)
    
    assert isinstance(mesh_data, MeshData)
    assert mesh_data.vertex_count == 0
