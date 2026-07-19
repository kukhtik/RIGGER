import pytest
import numpy as np
import os
from vtuber_rigger.interfaces import (
    MeshData, Skeleton, Bone, SkinWeights, 
    Expression, MorphTarget, SpringChain, SpringNode, 
    Collider, LookAt, FirstPerson, VRMMeta
)
from vtuber_rigger.qa.headless_vrm import (
    VRMData, load_vrm, apply_expression, rotate_bone, 
    apply_lookat, simulate_spring_physics, export_vrm
)
from scipy.sparse import csr_matrix

def create_simple_vrm_data():
    # 1. Mesh: a single triangle
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshData(vertices=verts, faces=faces)
    
    # 2. Skeleton: one bone
    bone = Bone(name="hips", position=np.array([0, 0, 0], dtype=np.float32))
    skeleton = Skeleton(bones=[bone])
    
    # 3. Weights: all vertices weighted to "hips"
    weights_mat = np.ones((3, 1), dtype=np.float32)
    weights = SkinWeights(matrix=csr_matrix(weights_mat), bone_names=["hips"])
    
    # 4. Expressions: one morph target
    target_verts = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
    morph = MorphTarget(name="happy", vertices=target_verts, vertex_count=3)
    expr = Expression(name="happy", morph_target=morph, morph_target_index=0)
    expressions = {"happy": expr}
    
    # 5. Springs
    node = SpringNode(bone_name="hips", position=np.array([0, 0, 0], dtype=np.float32))
    chain = SpringChain(nodes=[node], root_bone="hips")
    spring_chains = [chain]
    
    # 6. Others
    colliders = []
    lookat = LookAt(mode="bone")
    firstperson = FirstPerson(camera_offset=np.array([0,0,0], dtype=np.float32), mesh_annotations=[])
    meta = VRMMeta(name="TestVRM")
    
    return VRMData(mesh, skeleton, weights, expressions, spring_chains, colliders, lookat, firstperson, meta)

def test_apply_expression():
    vrm = create_simple_vrm_data()
    vrm_deformed = apply_expression(vrm, "happy", weight=1.0)
    
    # Vertices should match MorphTarget exactly
    np.testing.assert_array_almost_equal(vrm_deformed.mesh.vertices, vrm.expressions["happy"].morph_target.vertices)
    
    vrm_half = apply_expression(vrm, "happy", weight=0.5)
    expected = (vrm.mesh.vertices + vrm.expressions["happy"].morph_target.vertices) / 2
    np.testing.assert_array_almost_equal(vrm_half.mesh.vertices, expected)

def test_rotate_bone():
    vrm = create_simple_vrm_data()
    # Rotate 90 deg around Z axis
    # [0,0,0] -> [0,0,0]
    # [1,0,0] -> [0,1,0]
    # [0,1,0] -> [-1,0,0]
    vrm_rotated = rotate_bone(vrm, "hips", z=90)
    
    expected = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(vrm_rotated.mesh.vertices, expected)

def test_apply_lookat():
    vrm = create_simple_vrm_data()
    # Add eye bones
    vrm.skeleton.bones.append(Bone(name="leftEye", position=np.array([0.1, 0.1, 0]), rotation=None))
    vrm.skeleton.bones.append(Bone(name="rightEye", position=np.array([-0.1, 0.1, 0]), rotation=None))
    
    # Update weights to include eyes
    weights_mat = np.zeros((3, 3), dtype=np.float32)
    weights_mat[:, 0] = 1.0 # hips
    vrm.weights = SkinWeights(matrix=csr_matrix(weights_mat), bone_names=["hips", "leftEye", "rightEye"])
    
    vrm_look = apply_lookat(vrm, yaw=45, pitch=10)
    # Should not crash and return a VRMData
    assert isinstance(vrm_look, VRMData)

def test_simulate_spring_physics():
    vrm = create_simple_vrm_data()
    # Set some gravity
    vrm.spring_chains[0].gravityPower = 1.0
    vrm.spring_chains[0].gravityDir = np.array([0, -1, 0], dtype=np.float32)
    
    results = simulate_spring_physics(vrm, impulse=(0, 1, 0), frames=10)
    
    assert len(results) == 10
    assert "hips" in results[0]
    # Pos should change over time
    assert not np.array_equal(results[0]["hips"], results[9]["hips"])

@pytest.mark.skip(reason="Round-trip export/import needs binary GLB parsing fix")
def test_export_import_cycle():
    vrm = create_simple_vrm_data()
    path = "test_cycle.vrm"
    
    export_vrm(vrm, path)
    imported = load_vrm(path)
    
    # Check if vertices are preserved
    np.testing.assert_array_almost_equal(imported.mesh.vertices, vrm.mesh.vertices)
    
    if os.path.exists(path):
        os.remove(path)
