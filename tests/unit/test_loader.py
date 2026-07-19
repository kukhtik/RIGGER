"""Unit tests for Stage 1: Mesh Loader (vtuber_rigger.mesh.loader)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from vtuber_rigger.mesh.loader import (
    IMPORTS_OK,
    _build_adjacency,
    _ensure_triangles,
    _merge_duplicates,
    _normalize,
    _validate_mesh,
    load_mesh,
)
from vtuber_rigger.interfaces import MeshData


# ─── Helpers ────────────────────────────────────────────────────

def make_trimesh_cube() -> trimesh.Trimesh:
    """Return a subdivided cube with >100 vertices, 12+ triangular faces."""
    cube = trimesh.creation.box()
    # Subdivide to get >100 vertices (M1-4)
    cube = cube.subdivide()
    cube = cube.subdivide()
    cube = cube.subdivide()
    assert cube.faces.shape[1] == 3  # already triangles
    return cube


def write_tmp_file(mesh: trimesh.Trimesh, suffix: str = ".glb") -> Path:
    """Export a trimesh to a temp file and return its path."""
    path = Path(tempfile.mktemp(suffix=suffix))
    mesh.export(str(path), file_type=suffix.lstrip("."))
    return path


def cube_mesh_data() -> MeshData:
    """Return normalized MeshData built from a unit cube."""
    cube = make_trimesh_cube()
    verts = np.asarray(cube.vertices, dtype=np.float32)
    faces = np.asarray(cube.faces, dtype=np.int32)
    verts = _normalize(verts)
    return MeshData(
        vertices=verts,
        faces=faces,
        normals=np.asarray(cube.vertex_normals, dtype=np.float32),
        uvs=None,
        material_indices=None,
    )


# ─── Test normalize ──────────────────────────────────────────────

class TestNormalize:
    @pytest.mark.skip(reason="Test invariant is wrong: box 0.67×1.33×2 normalizes to range [-0.33, 0.33] not [-1, 1]")
    def test_centered_at_origin(self):
        # Box offset from origin should be shifted to center
        box = trimesh.creation.box(extents=[2.0, 4.0, 6.0])
        verts = np.asarray(box.vertices, dtype=np.float32)
        normed = _normalize(verts)
        assert np.allclose(normed.min(axis=0), -1.0, atol=1e-6)
        assert np.allclose(normed.max(axis=0), 1.0, atol=1e-6)

    def test_scale_reduces_extent_to_fit(self):
        box = trimesh.creation.box(extents=[100.0, 200.0, 300.0])
        verts = np.asarray(box.vertices, dtype=np.float32)
        normed = _normalize(verts)
        extents = normed.max(axis=0) - normed.min(axis=0)
        assert np.all(extents <= 2.0 + 1e-6)

    def test_already_normalized_unchanged(self):
        verts = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
        normed = _normalize(verts)
        assert np.allclose(normed, verts)


# ─── Test _ensure_triangles ─────────────────────────────────────

class TestEnsureTriangles:
    def test_already_triangles_returns_unchanged(self):
        verts = np.zeros((4, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        v_out, f_out = _ensure_triangles(verts, faces)
        assert f_out.shape == (2, 3)
        assert np.array_equal(f_out, faces)

    def test_quad_becomes_two_triangles(self):
        verts = np.zeros((4, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2, 3]], dtype=np.int32)  # quad
        v_out, f_out = _ensure_triangles(verts, faces)
        assert f_out.shape[1] == 3
        assert f_out.shape[0] == 2  # 1 quad → 2 triangles

    def test_ngon_fans_correctly(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 2, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)  # pentagon
        v_out, f_out = _ensure_triangles(verts, faces)
        assert f_out.shape[1] == 3
        assert f_out.shape[0] == 3  # 1 pentagon → 3 triangles


# ─── Test _merge_duplicates ─────────────────────────────────────

class TestMergeDuplicates:
    def test_no_duplicates_unchanged(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        v_out, f_out = _merge_duplicates(verts, faces)
        assert len(v_out) == 3
        assert np.array_equal(f_out, faces)

    def test_exact_duplicates_merged(self):
        verts = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        v_out, f_out = _merge_duplicates(verts, faces)
        assert len(v_out) == 2
        # Face should map to the surviving index
        assert np.all(f_out >= 0)
        assert f_out.shape[1] == 3

    def test_degenerate_faces_removed(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 0, 0], [0, 1, 2]], dtype=np.int32)  # first is degenerate
        v_out, f_out = _merge_duplicates(verts, faces)
        # Degenerate face [0,0,0] should be removed, leaving 1 valid face
        assert len(f_out) == 1
        assert f_out[0, 0] != f_out[0, 1] and f_out[0, 0] != f_out[0, 2]


# ─── Test _validate_mesh ────────────────────────────────────────

class TestValidateMesh:
    def test_valid_mesh_passes(self):
        md = cube_mesh_data()
        _validate_mesh(md.vertices, md.faces, md.normals, md.uvs, md.material_indices)
        # No exception = pass

    def test_m1_1_nan_vertex_raises(self):
        verts = np.array([[0, 0, 0], [np.nan, 0, 0], [1, 1, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        with pytest.raises(ValueError, match="M1-1"):
            _validate_mesh(verts, faces, None, None, None)

    def test_m1_1_inf_vertex_raises(self):
        verts = np.array([[0, 0, 0], [np.inf, 0, 0], [1, 1, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        with pytest.raises(ValueError, match="M1-1"):
            _validate_mesh(verts, faces, None, None, None)

    def test_m1_2_non_triangle_raises(self):
        verts = np.zeros((4, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2, 3]], dtype=np.int32)  # quad
        with pytest.raises(ValueError, match="M1-2"):
            _validate_mesh(verts, faces, None, None, None)

    def test_m1_3_out_of_range_index_raises(self):
        verts = np.zeros((3, 3), dtype=np.float32)
        faces = np.array([[0, 1, 99]], dtype=np.int32)  # index 99 out of range
        with pytest.raises(ValueError, match="M1-3"):
            _validate_mesh(verts, faces, None, None, None)

    def test_m1_3_negative_index_raises(self):
        verts = np.zeros((3, 3), dtype=np.float32)
        faces = np.array([[0, 1, -1]], dtype=np.int32)
        with pytest.raises(ValueError, match="M1-3"):
            _validate_mesh(verts, faces, None, None, None)

    def test_m1_4_too_few_vertices_raises(self):
        verts = np.zeros((99, 3), dtype=np.float32)
        faces = np.zeros((50, 3), dtype=np.int32)  # 50 < 100 vertices
        with pytest.raises(ValueError, match="M1-4"):
            _validate_mesh(verts, faces, None, None, None)

    def test_m1_5_bounding_box_exceeds_raises(self):
        # Create a subdivided cube that is not normalized (>100 verts)
        cube = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
        cube = cube.subdivide()
        cube = cube.subdivide()
        cube = cube.subdivide()
        verts = np.asarray(cube.vertices, dtype=np.float32)  # not normalized
        faces = np.asarray(cube.faces, dtype=np.int32)
        with pytest.raises(ValueError, match="M1-5"):
            _validate_mesh(verts, faces, None, None, None)

    def test_m1_6_nan_uv_raises(self):
        # Use 100+ vertices
        verts = np.zeros((101, 3), dtype=np.float32)
        for i in range(101):
            verts[i] = [i * 0.01, 0, 0]
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        uvs = np.zeros((101, 2), dtype=np.float32)
        uvs[1, 0] = np.nan
        with pytest.raises(ValueError, match="M1-6"):
            _validate_mesh(verts, faces, None, uvs, None)

    def test_m1_7_negative_material_index_raises(self):
        verts = np.zeros((101, 3), dtype=np.float32)
        for i in range(101):
            verts[i] = [i * 0.01, 0, 0]
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mat_idx = np.array([-1], dtype=np.int32)
        with pytest.raises(ValueError, match="M1-7"):
            _validate_mesh(verts, faces, None, None, mat_idx)

    def test_m1_8_duplicates_raises(self):
        # Create 101 vertices with a duplicate to pass M1-4 and trigger M1-8
        verts = np.zeros((101, 3), dtype=np.float32)
        verts[0] = [0, 0, 0]
        verts[1] = [0, 0, 0]  # duplicate of vert 0
        for i in range(2, 101):
            verts[i] = [i * 0.01, 0, 0]
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        with pytest.raises(ValueError, match="M1-8"):
            _validate_mesh(verts, faces, None, None, None)


# ─── Test _build_adjacency ───────────────────────────────────────

class TestBuildAdjacency:
    def test_cube_adjacency_correct_count(self):
        cube = make_trimesh_cube()
        verts = np.asarray(cube.vertices, dtype=np.float32)
        faces = np.asarray(cube.faces, dtype=np.int32)
        adj = _build_adjacency(verts, faces)
        n = len(verts)
        assert adj.shape == (n, n)
        assert adj.diagonal().sum() == 0  # no self-loops
        # Each vertex of a cube is adjacent to 3 others
        degrees = np.array(adj.sum(axis=1)).ravel()
        assert all(d >= 1 for d in degrees)

    def test_disconnected_vertices_have_no_edges(self):
        # 4 vertices: face [0,1,2] connects first 3, vertex 3 is isolated
        verts = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)  # only one triangle
        adj = _build_adjacency(verts, faces)
        # Vertex 3 is not connected to vertices 0, 1, or 2
        assert adj[3, 0] == 0 and adj[0, 3] == 0
        assert adj[3, 1] == 0 and adj[1, 3] == 0
        assert adj[3, 2] == 0 and adj[2, 3] == 0
        # But 0-1, 0-2, 1-2 ARE connected (via the face)
        assert adj[0, 1] > 0 and adj[1, 0] > 0


# ─── Test load_mesh (full pipeline) ─────────────────────────────

class TestLoadMeshGLB:
    def test_load_unit_cube_glb(self):
        cube = make_trimesh_cube()
        path = write_tmp_file(cube, ".glb")
        md = load_mesh(str(path))
        assert isinstance(md, MeshData)
        assert md.vertex_count >= 8
        assert md.face_count >= 12
        assert np.all(np.isfinite(md.vertices))
        assert np.all(np.isfinite(md.faces))
        assert np.all((md.faces >= 0) & (md.faces < md.vertex_count))
        amin, amax = md.bounding_box
        assert np.all(amin >= -1.0) and np.all(amax <= 1.0)
        path.unlink(missing_ok=True)

    def test_load_unit_cube_gltf(self):
        cube = make_trimesh_cube()
        path = write_tmp_file(cube, ".gltf")
        md = load_mesh(str(path))
        assert isinstance(md, MeshData)
        path.unlink(missing_ok=True)

    def test_load_unit_cube_obj(self):
        cube = make_trimesh_cube()
        path = write_tmp_file(cube, ".obj")
        md = load_mesh(str(path))
        assert isinstance(md, MeshData)
        path.unlink(missing_ok=True)

    def test_load_unit_cube_vrm(self):
        # VRM is glTF-based, export as .glb then rename to .vrm
        cube = make_trimesh_cube()
        path = write_tmp_file(cube, ".glb")
        # Rename to .vrm to test extension handling
        vrm_path = path.with_suffix(".vrm")
        path.rename(vrm_path)
        md = load_mesh(str(vrm_path))
        assert isinstance(md, MeshData)
        vrm_path.unlink(missing_ok=True)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_mesh("/nonexistent/path/to/model.glb")

    def test_unsupported_extension_raises(self):
        # Create a real file with unsupported extension
        path = Path(tempfile.mktemp(suffix=".bpx"))
        path.write_bytes(b"dummy")
        with pytest.raises((ValueError, FileNotFoundError)):
            load_mesh(str(path))
        path.unlink(missing_ok=True)

    def test_returned_mesh_has_required_fields(self):
        cube = make_trimesh_cube()
        path = write_tmp_file(cube)
        md = load_mesh(str(path))
        assert hasattr(md, "vertices")
        assert hasattr(md, "faces")
        assert hasattr(md, "normals")
        assert hasattr(md, "uvs")
        assert hasattr(md, "material_indices")
        path.unlink(missing_ok=True)

    def test_vertices_are_float32(self):
        cube = make_trimesh_cube()
        path = write_tmp_file(cube)
        md = load_mesh(str(path))
        assert md.vertices.dtype == np.float32
        assert md.faces.dtype == np.int32
        path.unlink(missing_ok=True)

    def test_bounding_box_within_unit_cube(self):
        # Non-unit mesh gets normalized to [-1,1]³
        sphere = trimesh.creation.icosphere(subdivisions=2)
        path = write_tmp_file(sphere, ".glb")
        md = load_mesh(str(path))
        amin, amax = md.bounding_box
        assert np.all(amin >= -1.0)
        assert np.all(amax <= 1.0)
        path.unlink(missing_ok=True)

    def test_adjacency_matrix_shape(self):
        cube = make_trimesh_cube()
        path = write_tmp_file(cube)
        md = load_mesh(str(path))
        # Adjacency is computed but not stored in MeshData
        # Verify the function works directly
        adj = _build_adjacency(md.vertices, md.faces)
        assert adj.shape[0] == adj.shape[1] == md.vertex_count
        path.unlink(missing_ok=True)


class TestFBXLoading:
    def test_fbx_requires_blender(self, monkeypatch):
        """FBX loading should raise RuntimeError when Blender is unavailable."""
        import os
        # Create a dummy .fbx file so the file-exists check passes
        fbx_path = Path(tempfile.mktemp(suffix=".fbx"))
        fbx_path.write_bytes(b"dummy fbx content")
        monkeypatch.setenv("BLENDER_PATH", "/nonexistent/blender")
        from vtuber_rigger.mesh import loader as loader_module
        monkeypatch.setattr(loader_module, "_find_blender", lambda: None)
        try:
            with pytest.raises(RuntimeError, match="Blender not found"):
                loader_module.load_mesh(str(fbx_path))
        finally:
            fbx_path.unlink(missing_ok=True)


class TestSentinel:
    def test_imports_ok_true(self):
        assert IMPORTS_OK is True
