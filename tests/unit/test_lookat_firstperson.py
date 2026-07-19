"""Unit tests for Stage 7: LookAt + FirstPerson (lookat_firstperson.py)."""

from __future__ import annotations

import numpy as np
import pytest

from vtuber_rigger.interfaces import (
    Bone,
    LookAt,
    FirstPerson,
    MeshData,
    Skeleton,
)
from vtuber_rigger.rig.lookat_firstperson import (
    create_lookat,
    create_firstperson,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_simple_skeleton(with_eyes: bool = False) -> Skeleton:
    """Minimal valid skeleton for testing."""
    bones = [
        Bone(name="hips", position=np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        Bone(name="spine", position=np.array([0.0, 0.2, 0.0], dtype=np.float32), parent="hips"),
        Bone(name="chest", position=np.array([0.0, 0.4, 0.0], dtype=np.float32), parent="spine"),
        Bone(name="head", position=np.array([0.0, 0.6, 0.0], dtype=np.float32), parent="chest"),
        Bone(
            name="leftUpperArm",
            position=np.array([-0.2, 0.4, 0.0], dtype=np.float32),
            parent="chest",
        ),
        Bone(
            name="leftLowerArm",
            position=np.array([-0.4, 0.4, 0.0], dtype=np.float32),
            parent="leftUpperArm",
        ),
        Bone(
            name="leftHand",
            position=np.array([-0.5, 0.4, 0.0], dtype=np.float32),
            parent="leftLowerArm",
        ),
        Bone(
            name="rightUpperArm",
            position=np.array([0.2, 0.4, 0.0], dtype=np.float32),
            parent="chest",
        ),
        Bone(
            name="rightLowerArm",
            position=np.array([0.4, 0.4, 0.0], dtype=np.float32),
            parent="rightUpperArm",
        ),
        Bone(
            name="rightHand",
            position=np.array([0.5, 0.4, 0.0], dtype=np.float32),
            parent="rightLowerArm",
        ),
        Bone(name="leftUpperLeg", position=np.array([-0.1, -0.2, 0.0], dtype=np.float32), parent="hips"),
        Bone(
            name="leftLowerLeg",
            position=np.array([-0.1, -0.5, 0.0], dtype=np.float32),
            parent="leftUpperLeg",
        ),
        Bone(
            name="leftFoot",
            position=np.array([-0.1, -0.8, 0.0], dtype=np.float32),
            parent="leftLowerLeg",
        ),
        Bone(name="rightUpperLeg", position=np.array([0.1, -0.2, 0.0], dtype=np.float32), parent="hips"),
        Bone(
            name="rightLowerLeg",
            position=np.array([0.1, -0.5, 0.0], dtype=np.float32),
            parent="rightUpperLeg",
        ),
        Bone(
            name="rightFoot",
            position=np.array([0.1, -0.8, 0.0], dtype=np.float32),
            parent="rightLowerLeg",
        ),
    ]
    if with_eyes:
        bones.append(
            Bone(
                name="leftEye",
                position=np.array([-0.06, 0.65, 0.08], dtype=np.float32),
                parent="head",
            )
        )
        bones.append(
            Bone(
                name="rightEye",
                position=np.array([0.06, 0.65, 0.08], dtype=np.float32),
                parent="head",
            )
        )
    return Skeleton(bones=bones)


def _make_simple_mesh() -> MeshData:
    """Minimal valid MeshData."""
    # Simple box mesh
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )
    normals = np.zeros((8, 3), dtype=np.float32)
    return MeshData(vertices=vertices, faces=faces, normals=normals)


# ─── create_lookat tests ───────────────────────────────────────────────────────

class TestCreateLookAt:
    """M7-1..M7-5 invariants."""

    def test_mode_bone_when_eyes_present(self):
        skel = _make_simple_skeleton(with_eyes=True)
        lookat = create_lookat(skel)
        assert lookat.mode == "bone"

    def test_mode_expression_when_no_eyes(self):
        skel = _make_simple_skeleton(with_eyes=False)
        lookat = create_lookat(skel)
        assert lookat.mode == "expression"

    def test_yaw_range_finite(self):
        skel = _make_simple_skeleton(with_eyes=False)
        lookat = create_lookat(skel)
        assert np.isfinite(lookat.yaw_range)
        assert lookat.yaw_range == pytest.approx(1.5708, abs=1e-4)

    def test_pitch_range_finite(self):
        skel = _make_simple_skeleton(with_eyes=False)
        lookat = create_lookat(skel)
        assert np.isfinite(lookat.pitch_range)
        assert lookat.pitch_range == pytest.approx(1.5708, abs=1e-4)

    def test_returns_lookat_instance(self):
        skel = _make_simple_skeleton(with_eyes=True)
        lookat = create_lookat(skel)
        assert isinstance(lookat, LookAt)

    def test_yaw_and_pitch_ranges_default_90deg(self):
        """M7-4 M7-5: default ranges ~90 degrees."""
        skel = _make_simple_skeleton(with_eyes=False)
        lookat = create_lookat(skel)
        assert lookat.yaw_range == pytest.approx(np.pi / 2, abs=1e-4)
        assert lookat.pitch_range == pytest.approx(np.pi / 2, abs=1e-4)


# ─── create_firstperson tests ─────────────────────────────────────────────────

class TestCreateFirstPerson:
    """M7-6 M7-7 invariants."""

    def test_camera_offset_finite(self):
        skel = _make_simple_skeleton(with_eyes=True)
        mesh = _make_simple_mesh()
        fp = create_firstperson(skel, mesh)
        assert np.all(np.isfinite(fp.camera_offset))

    def test_camera_offset_y_in_range(self):
        """Camera offset y should be in [-1, 0]."""
        skel = _make_simple_skeleton(with_eyes=True)
        mesh = _make_simple_mesh()
        fp = create_firstperson(skel, mesh)
        assert -1.0 <= fp.camera_offset[1] <= 0.0

    def test_mesh_annotations_not_empty(self):
        skel = _make_simple_skeleton(with_eyes=True)
        mesh = _make_simple_mesh()
        fp = create_firstperson(skel, mesh)
        assert len(fp.mesh_annotations) >= 1

    def test_mesh_annotation_types_valid(self):
        """M7-7: annotation type ∈ {auto, both, thirdPerson, firstPerson}."""
        skel = _make_simple_skeleton(with_eyes=True)
        mesh = _make_simple_mesh()
        fp = create_firstperson(skel, mesh)
        valid_types = {"auto", "both", "thirdPerson", "firstPerson"}
        for ann in fp.mesh_annotations:
            assert ann["type"] in valid_types, f"Invalid type: {ann['type']}"

    def test_returns_firstperson_instance(self):
        skel = _make_simple_skeleton(with_eyes=True)
        mesh = _make_simple_mesh()
        fp = create_firstperson(skel, mesh)
        assert isinstance(fp, FirstPerson)

    def test_works_without_eyes(self):
        """firstPerson should work even without eye bones."""
        skel = _make_simple_skeleton(with_eyes=False)
        mesh = _make_simple_mesh()
        fp = create_firstperson(skel, mesh)
        assert isinstance(fp, FirstPerson)
        assert np.all(np.isfinite(fp.camera_offset))
        assert len(fp.mesh_annotations) >= 1

    def test_camera_offset_3d_vector(self):
        """M7-6: camera offset is a 3D vector."""
        skel = _make_simple_skeleton(with_eyes=True)
        mesh = _make_simple_mesh()
        fp = create_firstperson(skel, mesh)
        assert fp.camera_offset.shape == (3,)
