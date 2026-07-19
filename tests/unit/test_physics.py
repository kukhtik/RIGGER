import numpy as np
import pytest
from vtuber_rigger.qa.physics import simulate_impulse_response, simulate_collision_test, CollisionTestResult
from vtuber_rigger.interfaces import Skeleton, Bone, SpringChain, SpringNode, Collider, MeshData, VRMMeta
from vtuber_rigger.qa.headless_vrm import VRMData

@pytest.fixture
def mock_vrm_path():
    return "dummy.vrm"

@pytest.fixture
def mock_vrm_data():
    # Create a simple VRM with one spring chain and one collider
    bone = Bone(name="head", position=np.array([0, 0, 0], dtype=np.float32))
    skeleton = Skeleton(bones=[bone])
    
    # Spring chain: root "head", tip at (0, 0.1, 0)
    nodes = [
        SpringNode(bone_name="head", position=np.array([0, 0, 0], dtype=np.float32)),
        SpringNode(bone_name="hair_tip", position=np.array([0, 0.1, 0], dtype=np.float32))
    ]
    chain = SpringChain(
        nodes=nodes,
        root_bone="head",
        stiffness=10.0, # Increased stiffness for faster oscillation
        dragForce=0.5,
        hitRadius=0.01,
        gravityPower=0.0, 
        gravityDir=np.array([0, -1, 0], dtype=np.float32)
    )
    
    collider = Collider(center=np.array([0, 0.05, 0], dtype=np.float32), radius=0.02)
    
    mesh = MeshData(vertices=np.zeros((10, 3), dtype=np.float32), faces=np.zeros((0, 3), dtype=np.int32))
    meta = VRMMeta(name="TestVRM")
    
    return VRMData(
        mesh=mesh,
        skeleton=skeleton,
        weights=None,
        expressions={},
        spring_chains=[chain],
        colliders=[collider],
        lookat=None,
        firstperson=None,
        meta=meta
    )

def test_oscillation_frequency(mock_vrm_path, mock_vrm_data, monkeypatch):
    monkeypatch.setattr("vtuber_rigger.qa.physics.load_vrm", lambda path: mock_vrm_data)
    
    # Apply impulse in X
    res = simulate_impulse_response(mock_vrm_path, impulse=(1.0, 0, 0), duration=1.0, fps=60)
    
    # The response should be an oscillating signal in X
    x_coords = res[:, 0]
    
    # Find zero crossings to estimate frequency
    crossings = []
    for i in range(len(x_coords) - 1):
        if x_coords[i] * x_coords[i+1] < 0:
            crossings.append(i)
            
    assert len(crossings) >= 1, "Should see at least one zero crossing in 1 second"

def test_damping_and_settling_time(mock_vrm_path, mock_vrm_data, monkeypatch):
    monkeypatch.setattr("vtuber_rigger.qa.physics.load_vrm", lambda path: mock_vrm_data)
    
    # Simulate longer duration to check settling
    res = simulate_impulse_response(mock_vrm_path, impulse=(1.0, 0, 0), duration=3.0, fps=60)
    
    # Tip should return towards rest position (0, 0.1, 0)
    final_pos = res[-1]
    rest_pos = np.array([0, 0.1, 0])
    
    dist = np.linalg.norm(final_pos - rest_pos)
    assert dist < 0.1, f"Should have settled near rest position, got dist {dist}"

def test_collision_deflection(mock_vrm_path, mock_vrm_data, monkeypatch):
    monkeypatch.setattr("vtuber_rigger.qa.physics.load_vrm", lambda path: mock_vrm_data)
    
    result = simulate_collision_test(mock_vrm_path)
    
    dist_to_center = np.linalg.norm(result.final_position - np.array([0, 0.05, 0]))
    
    assert dist_to_center >= 0.029, f"Tip should be deflected by collider, got dist {dist_to_center}"
    assert result.min_distance >= -0.001, "Min distance should not be deeply negative"

def test_no_spring_bones(mock_vrm_path, monkeypatch):
    empty_vrm = VRMData(
        mesh=MeshData(np.zeros((0,3)), np.zeros((0,3))),
        skeleton=Skeleton([]),
        weights=None,
        expressions={},
        spring_chains=[],
        colliders=[],
        lookat=None,
        firstperson=None,
        meta=VRMMeta("Empty")
    )
    monkeypatch.setattr("vtuber_rigger.qa.physics.load_vrm", lambda path: empty_vrm)
    
    res = simulate_impulse_response(mock_vrm_path)
    assert res.shape == (0, 3)
    
    col_res = simulate_collision_test(mock_vrm_path)
    assert col_res.min_distance == float('inf')
