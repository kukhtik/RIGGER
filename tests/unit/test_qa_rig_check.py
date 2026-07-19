\
"""Unit tests for vtuber_rigger.qa.rig_check."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.sparse import csr_matrix

from vtuber_rigger.interfaces import (
    REQUIRED_BONES,
    Bone,
    MeshData,
    Skeleton,
    SkinWeights,
)
from vtuber_rigger.qa.rig_check import check_rig


# ─── Fixtures ───────────────────────────────────────────────────────────────

def make_sphere_mesh(n: int = 3) -> MeshData:
    """Normalized ico-sphere as test mesh."""
    sphere = trimesh.creation.icosphere(subdivisions=n)
    v = np.asarray(sphere.vertices, dtype=np.float32)
    center = v.min(axis=0) + (v.max(axis=0) - v.min(axis=0)) / 2
    scale = 2.0 / max((v.max(axis=0) - v.min(axis=0)).max(), 1e-10)
    v = (v - center) * scale
    f = np.asarray(sphere.faces, dtype=np.int32)
    normals = np.asarray(trimesh.Trimesh(vertices=v, faces=f).vertex_normals, dtype=np.float32)
    return MeshData(vertices=v, faces=f, normals=normals)


def valid_skeleton() -> Skeleton:
    """A complete valid VRM skeleton with all 15 required bones."""
    bones = [
        Bone("hips",         position=np.array([0.0,  0.9,  0.0],   dtype=np.float32)),
        Bone("spine",        position=np.array([0.0,  1.0,  0.0],   dtype=np.float32), parent="hips"),
        Bone("head",         position=np.array([0.0,  1.3,  0.0],   dtype=np.float32), parent="spine"),
        Bone("leftUpperArm", position=np.array([-0.2, 1.1,  0.0],  dtype=np.float32), parent="spine"),
        Bone("leftLowerArm", position=np.array([-0.4, 0.9,  0.0],  dtype=np.float32), parent="leftUpperArm"),
        Bone("leftHand",     position=np.array([-0.5, 0.75, 0.0],  dtype=np.float32), parent="leftLowerArm"),
        Bone("rightUpperArm",position=np.array([ 0.2, 1.1,  0.0],  dtype=np.float32), parent="spine"),
        Bone("rightLowerArm",position=np.array([ 0.4, 0.9,  0.0],  dtype=np.float32), parent="rightUpperArm"),
        Bone("rightHand",    position=np.array([ 0.5, 0.75, 0.0],  dtype=np.float32), parent="rightLowerArm"),
        Bone("leftUpperLeg", position=np.array([-0.1, 0.5,  0.0],  dtype=np.float32), parent="hips"),
        Bone("leftLowerLeg", position=np.array([-0.1, 0.25, 0.0],  dtype=np.float32), parent="leftUpperLeg"),
        Bone("leftFoot",     position=np.array([-0.1, 0.0,  0.0],  dtype=np.float32), parent="leftLowerLeg"),
        Bone("rightUpperLeg",position=np.array([ 0.1, 0.5,  0.0], dtype=np.float32), parent="hips"),
        Bone("rightLowerLeg",position=np.array([ 0.1, 0.25, 0.0], dtype=np.float32), parent="rightUpperLeg"),
        Bone("rightFoot",    position=np.array([ 0.1, 0.0,  0.0], dtype=np.float32), parent="rightLowerLeg"),
    ]
    return Skeleton(bones=bones)


def uniform_weights(mesh: MeshData, bone_names: list[str]) -> SkinWeights:
    """
    Build a sparse weight matrix with at most 4 bone influences per vertex.
    """
    n_verts = mesh.vertex_count
    n_bones = len(bone_names)
    # Use at most 4 bones per vertex to satisfy M4-4 (<= 8)
    k = min(4, n_bones)
    data = []
    row = []
    col = []
    for v in range(n_verts):
        w = 1.0 / k
        for b in range(k):
            data.append(w)
            row.append(v)
            col.append(b)
    matrix = csr_matrix(
        (data, (row, col)),
        shape=(n_verts, n_bones),
        dtype=np.float32,
    )
    return SkinWeights(matrix=matrix, bone_names=bone_names)


# ─── Valid rig tests ────────────────────────────────────────────────────────

class TestValidRig:

    def test_valid_skeleton_and_weights(self):
        mesh = make_sphere_mesh()
        skel = valid_skeleton()
        bone_names = [b.name for b in skel.bones]
        from vtuber_rigger.rig.weights import compute_weights
        weights = compute_weights(mesh, skel)
        errors = check_rig(skel, weights, mesh)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_extra_bones_also_valid(self):
        """Skeletons with extra bones beyond the 15 required are valid."""
        mesh = make_sphere_mesh()
        skel = valid_skeleton()
        # Add optional bones
        skel.bones.append(Bone("chest",      position=np.array([0.0, 1.05, 0.0], dtype=np.float32), parent="spine"))
        skel.bones.append(Bone("leftEye",    position=np.array([-0.05, 1.32, 0.1], dtype=np.float32), parent="head"))
        bone_names = [b.name for b in skel.bones]
        from vtuber_rigger.rig.weights import compute_weights
        weights = compute_weights(mesh, skel)
        errors = check_rig(skel, weights, mesh)
        assert errors == [], f"expected no errors, got: {errors}"


# ─── Invalid rig tests ──────────────────────────────────────────────────────

class TestInvalidRig:

    def test_too_few_bones(self):
        """Skeleton with < 15 bones should fail."""
        mesh = make_sphere_mesh()
        bones = [
            Bone("hips", position=np.array([0.0, 0.9, 0.0], dtype=np.float32)),
            Bone("spine", position=np.array([0.0, 1.0, 0.0], dtype=np.float32), parent="hips"),
        ]
        skel = Skeleton(bones=bones)
        bone_names = [b.name for b in skel.bones]
        weights = uniform_weights(mesh, bone_names)
        errors = check_rig(skel, weights, mesh)
        assert any("15" in e for e in errors), f"expected bone count error, got: {errors}"

    def test_missing_required_bone(self):
        """Missing any required bone should fail."""
        mesh = make_sphere_mesh()
        skel = valid_skeleton()
        # Remove head bone
        skel.bones = [b for b in skel.bones if b.name != "head"]
        bone_names = [b.name for b in skel.bones]
        weights = uniform_weights(mesh, bone_names)
        errors = check_rig(skel, weights, mesh)
        assert any("head" in e for e in errors), f"expected missing bone error, got: {errors}"

    def test_cycle_in_skeleton(self):
        """A cycle in the parent chain should be detected."""
        mesh = make_sphere_mesh()
        bones = [
            # hips → spine → head → hips (cycle back to hips)
            Bone("hips",  position=np.array([0.0, 0.9, 0.0], dtype=np.float32), parent="head"),
            Bone("spine", position=np.array([0.0, 1.0, 0.0], dtype=np.float32), parent="hips"),
            Bone("head",  position=np.array([0.0, 1.3, 0.0], dtype=np.float32), parent="spine"),
        ]
        skel = Skeleton(bones=bones)
        bone_names = [b.name for b in skel.bones]
        weights = uniform_weights(mesh, bone_names)
        errors = check_rig(skel, weights, mesh)
        assert any("cycle" in e.lower() for e in errors), f"expected cycle error, got: {errors}"

    def test_non_normalized_weights(self):
        """Weights whose per-vertex sum != 1.0 should fail."""
        mesh = make_sphere_mesh()
        skel = valid_skeleton()
        bone_names = [b.name for b in skel.bones]
        n_verts = mesh.vertex_count
        n_bones = len(bone_names)
        # Double weights (sum to 2.0)
        data = []
        row = []
        col = []
        per_v = n_bones
        for v in range(n_verts):
            for b in range(per_v):
                data.append(2.0 / per_v)  # deliberately wrong
                row.append(v)
                col.append(b)
        matrix = csr_matrix((data, (row, col)), shape=(n_verts, n_bones), dtype=np.float32)
        weights = SkinWeights(matrix=matrix, bone_names=bone_names)
        errors = check_rig(skel, weights, mesh)
        assert any("normalized" in e or "sum" in e for e in errors), \
            f"expected normalization error, got: {errors}"

    def test_too_many_influences_per_vertex(self):
        """> 8 bone influences per vertex should fail M4-4."""
        mesh = make_sphere_mesh()
        skel = valid_skeleton()
        bone_names = [b.name for b in skel.bones]
        n_verts = mesh.vertex_count
        n_bones = len(bone_names)
        # Assign 12 bones per vertex (> 8)
        per_v = min(12, n_bones)
        data = []
        row = []
        col = []
        for v in range(n_verts):
            w = 1.0 / per_v
            for b in range(per_v):
                data.append(w)
                row.append(v)
                col.append(b % n_bones)
        matrix = csr_matrix((data, (row, col)), shape=(n_verts, n_bones), dtype=np.float32)
        weights = SkinWeights(matrix=matrix, bone_names=bone_names)
        errors = check_rig(skel, weights, mesh)
        assert any("8" in e for e in errors), f"expected sparsity error, got: {errors}"

    def test_unused_bone(self):
        """A bone with zero weighted vertices should fail M4-5."""
        mesh = make_sphere_mesh()
        skel = valid_skeleton()
        bone_names = [b.name for b in skel.bones]
        n_verts = mesh.vertex_count
        n_bones = len(bone_names)
        # Only use first bone
        per_v = 1
        data = []
        row = []
        col = []
        for v in range(n_verts):
            data.append(1.0)
            row.append(v)
            col.append(0)  # only bone 0
        matrix = csr_matrix((data, (row, col)), shape=(n_verts, n_bones), dtype=np.float32)
        weights = SkinWeights(matrix=matrix, bone_names=bone_names)
        errors = check_rig(skel, weights, mesh)
        assert any("unused" in e or "weight" in e.lower() for e in errors), \
            f"expected unused bone error, got: {errors}"

    def test_isolated_vertex(self):
        """A vertex with no bone weights should fail M4-6."""
        mesh = make_sphere_mesh()
        skel = valid_skeleton()
        bone_names = [b.name for b in skel.bones]
        n_verts = mesh.vertex_count
        n_bones = len(bone_names)
        # Normal weights for all but the last vertex
        per_v = min(4, n_bones)
        data = []
        row = []
        col = []
        for v in range(n_verts - 1):
            w = 1.0 / per_v
            for b in range(per_v):
                data.append(w)
                row.append(v)
                col.append(b % n_bones)
        matrix = csr_matrix((data, (row, col)), shape=(n_verts, n_bones), dtype=np.float32)
        weights = SkinWeights(matrix=matrix, bone_names=bone_names)
        errors = check_rig(skel, weights, mesh)
        assert any("isolated" in e or "no bone weight" in e.lower() for e in errors), \
            f"expected isolated vertex error, got: {errors}"
