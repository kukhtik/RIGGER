"""Unit tests for Stage 4: Skin Weights (vtuber_rigger.rig.weights)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.sparse import csr_matrix

from vtuber_rigger.interfaces import Bone, MeshData, Skeleton, SkinWeights
from vtuber_rigger.rig.skeleton import predict_skeleton
from vtuber_rigger.rig.weights import (
    IMPORTS_OK,
    _bone_segment_for_bone,
    _build_face_adjacency,
    _compute_bone_distances,
    _laplacian_smooth,
    _normalize_rows,
    _point_to_segment_distance,
    compute_weights,
    validate_weights,
)


# ─── Test mesh fixture ─────────────────────────────────────────────

def make_sphere_mesh_data(n_vertices: int = 200) -> MeshData:
    """
    Build a normalized triangle sphere mesh with `n_vertices`.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3)
    verts = np.asarray(sphere.vertices, dtype=np.float32)
    amin, amax = verts.min(axis=0), verts.max(axis=0)
    center = (amin + amax) / 2.0
    scale = 2.0 / max((amax - amin).max(), 1e-10)
    verts = (verts - center) * scale
    faces = np.asarray(sphere.faces, dtype=np.int32)
    normals = np.asarray(sphere.vertex_normals, dtype=np.float32)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)
    return MeshData(
        vertices=verts,
        faces=faces,
        normals=normals,
        uvs=None,
        material_indices=None,
    )


def make_skeleton_for_test() -> Skeleton:
    """Minimal but valid VRM skeleton for weight tests."""
    bones = [
        Bone(name="hips", position=np.array([0.0, -0.5, 0.0], dtype=np.float32), parent=None),
        Bone(name="spine", position=np.array([0.0, 0.0, 0.0], dtype=np.float32), parent="hips"),
        Bone(name="head", position=np.array([0.0, 0.5, 0.0], dtype=np.float32), parent="spine"),
        Bone(name="leftUpperArm", position=np.array([-0.3, 0.1, 0.0], dtype=np.float32), parent="spine"),
        Bone(name="leftLowerArm", position=np.array([-0.6, 0.0, 0.0], dtype=np.float32), parent="leftUpperArm"),
        Bone(name="leftHand", position=np.array([-0.8, -0.1, 0.0], dtype=np.float32), parent="leftLowerArm"),
        Bone(name="rightUpperArm", position=np.array([0.3, 0.1, 0.0], dtype=np.float32), parent="spine"),
        Bone(name="rightLowerArm", position=np.array([0.6, 0.0, 0.0], dtype=np.float32), parent="rightUpperArm"),
        Bone(name="rightHand", position=np.array([0.8, -0.1, 0.0], dtype=np.float32), parent="rightLowerArm"),
        Bone(name="leftUpperLeg", position=np.array([-0.1, -0.8, 0.0], dtype=np.float32), parent="hips"),
        Bone(name="leftLowerLeg", position=np.array([-0.1, -1.2, 0.0], dtype=np.float32), parent="leftUpperLeg"),
        Bone(name="leftFoot", position=np.array([-0.1, -1.5, 0.0], dtype=np.float32), parent="leftLowerLeg"),
        Bone(name="rightUpperLeg", position=np.array([0.1, -0.8, 0.0], dtype=np.float32), parent="hips"),
        Bone(name="rightLowerLeg", position=np.array([0.1, -1.2, 0.0], dtype=np.float32), parent="rightUpperLeg"),
        Bone(name="rightFoot", position=np.array([0.1, -1.5, 0.0], dtype=np.float32), parent="rightLowerLeg"),
    ]
    return Skeleton(bones=bones)


# ─── Test _point_to_segment_distance ───────────────────────────────

class TestPointToSegmentDistance:
    def test_point_at_segment_midpoint(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        d = _point_to_segment_distance(p, a, b)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_point_at_endpoint(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        d = _point_to_segment_distance(a, a, b)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_point_perpendicular(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        d = _point_to_segment_distance(p, a, b)
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_clamped_endpoint_near_a(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p = np.array([-0.5, 0.0, 0.0], dtype=np.float32)  # beyond a
        d = _point_to_segment_distance(p, a, b)
        assert d == pytest.approx(0.25, abs=1e-6)


# ─── Test _bone_segment_for_bone ──────────────────────────────────

class TestBoneSegmentForBone:
    def test_returns_two_points(self):
        skel = make_skeleton_for_test()
        a, b = _bone_segment_for_bone("hips", skel)
        assert a.shape == (3,)
        assert b.shape == (3,)

    def test_child_as_endpoint(self):
        skel = make_skeleton_for_test()
        _, b = _bone_segment_for_bone("hips", skel)
        spine = skel.get_bone("spine")
        assert np.allclose(b, spine.position)


# ─── Test _compute_bone_distances ────────────────────────────────

class TestComputeBoneDistances:
    def test_returns_array_of_correct_length(self):
        skel = make_skeleton_for_test()
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bone_names = [b.name for b in skel.bones]
        dists = _compute_bone_distances(v, skel, bone_names)
        assert len(dists) == len(bone_names)
        assert np.all(dists >= 0)

    def test_zero_distance_for_vertex_at_bone_position(self):
        skel = make_skeleton_for_test()
        bone_names = [b.name for b in skel.bones]
        v = skel.get_bone("head").position.copy()
        dists = _compute_bone_distances(v, skel, bone_names)
        # Nearest bone should be head (distance ~0)
        assert dists.min() < 0.01


# ─── Test _build_face_adjacency ───────────────────────────────────

class TestBuildFaceAdjacency:
    def test_symmetric(self):
        mesh = make_sphere_mesh_data()
        adj = _build_face_adjacency(mesh)
        assert np.allclose(adj.toarray(), adj.toarray().T)

    def test_diagonal_zero(self):
        mesh = make_sphere_mesh_data()
        adj = _build_face_adjacency(mesh)
        assert np.allclose(adj.diagonal(), 0)

    def test_no_negative_entries(self):
        mesh = make_sphere_mesh_data()
        adj = _build_face_adjacency(mesh)
        assert np.all(adj.toarray() >= 0)


# ─── Test _normalize_rows ─────────────────────────────────────────

class TestNormalizeRows:
    def test_row_sums_become_one(self):
        # Dense matrix with known row sums
        data = np.array([[2.0, 4.0], [1.0, 3.0], [0.5, 0.5]], dtype=np.float32)
        W = csr_matrix(data)
        W_norm = _normalize_rows(W)
        sums = np.array(W_norm.sum(axis=1)).ravel()
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_zero_row_unchanged(self):
        data = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        W = csr_matrix(data)
        W_norm = _normalize_rows(W)
        # First row stays all zeros (guard)
        assert np.allclose(W_norm.toarray()[0], 0.0)


# ─── Test _laplacian_smooth ────────────────────────────────────────

class TestLaplacianSmooth:
    def test_shape_unchanged(self):
        mesh = make_sphere_mesh_data()
        adj = _build_face_adjacency(mesh)
        n = mesh.vertex_count
        data = np.random.rand(n, 5).astype(np.float32)
        W = csr_matrix(data)
        W_smooth = _laplacian_smooth(W, adj, iters=3)
        assert W_smooth.shape == W.shape

    def test_result_is_sparse(self):
        mesh = make_sphere_mesh_data()
        adj = _build_face_adjacency(mesh)
        n = mesh.vertex_count
        data = np.random.rand(n, 5).astype(np.float32)
        W = csr_matrix(data)
        W_smooth = _laplacian_smooth(W, adj, iters=3)
        assert isinstance(W_smooth, csr_matrix)


# ─── Test compute_weights ──────────────────────────────────────────

class TestComputeWeights:
    def test_returns_skin_weights_instance(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        assert isinstance(sw, SkinWeights)

    def test_matrix_shape_matches(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        assert sw.matrix.shape[0] == mesh.vertex_count
        assert sw.matrix.shape[1] == len(skel.bones)

    def test_bone_names_match(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        assert sw.bone_names == [b.name for b in skel.bones]

    def test_matrix_is_sparse(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        assert isinstance(sw.matrix, csr_matrix)

    def test_row_sums_are_one(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        row_sums = np.array(sw.matrix.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 1.0, atol=1e-4)

    def test_no_negative_weights(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        assert np.all(sw.matrix.data >= 0)

    def test_no_weights_exceed_one(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        assert np.all(sw.matrix.data <= 1.0)

    def test_each_vertex_has_at_most_8_non_zero(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        nnz_per_row = np.diff(sw.matrix.indptr)
        assert np.all(nnz_per_row <= 8)

    def test_each_bone_is_used(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        W = sw.matrix
        any_used = np.array((W > 0).todense()).any(axis=0).ravel()
        unused = [sw.bone_names[i] for i in range(len(sw.bone_names)) if not any_used[i]]
        assert not unused, f"Unused bones: {unused}"

    def test_no_isolated_vertices(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        W = sw.matrix
        all_zero = np.where(np.array(W.sum(axis=1)).ravel() == 0)[0]
        assert len(all_zero) == 0, f"Isolated vertices: {all_zero}"

    def test_get_vertex_weights(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        for v_idx in range(min(5, mesh.vertex_count)):
            w = sw.get_vertex_weights(v_idx)
            assert isinstance(w, dict)
            # Weights should be positive
            assert all(wv > 0 for wv in w.values())
            # Weights should sum to ~1
            assert sum(w.values()) == pytest.approx(1.0, abs=1e-4)


# ─── Test validate_weights ────────────────────────────────────────

class TestValidateWeights:
    def test_no_errors_for_valid_output(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        errors = validate_weights(mesh, skel, sw)
        assert errors == [], f"M4 validation errors: {errors}"

    def test_detects_negative_weights(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        # Corrupt a value
        data = sw.matrix.data.copy()
        data[0] = -0.1
        sw_corrupt = SkinWeights(
            matrix=csr_matrix((data, sw.matrix.indices, sw.matrix.indptr), shape=sw.matrix.shape),
            bone_names=sw.bone_names,
        )
        errors = validate_weights(mesh, skel, sw_corrupt)
        assert any("M4-1" in e for e in errors)

    def test_detects_weights_over_one(self):
        mesh = make_sphere_mesh_data()
        skel = make_skeleton_for_test()
        sw = compute_weights(mesh, skel)
        data = sw.matrix.data.copy()
        data[0] = 1.5
        sw_corrupt = SkinWeights(
            matrix=csr_matrix((data, sw.matrix.indices, sw.matrix.indptr), shape=sw.matrix.shape),
            bone_names=sw.bone_names,
        )
        errors = validate_weights(mesh, skel, sw_corrupt)
        assert any("M4-2" in e for e in errors)


# ─── Test integration: predict_skeleton + compute_weights ──────────

class TestSkeletonWeightsIntegration:
    def test_full_pipeline(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        sw = compute_weights(mesh, skel)
        assert sw.matrix.shape[0] == mesh.vertex_count
        assert sw.matrix.shape[1] == len(skel.bones)
        errors = validate_weights(mesh, skel, sw)
        assert errors == [], f"Integration M4 errors: {errors}"


class TestSentinel:
    def test_imports_ok_true(self):
        assert IMPORTS_OK is True
