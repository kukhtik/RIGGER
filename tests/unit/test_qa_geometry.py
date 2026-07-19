\
"""Unit tests for vtuber_rigger.qa.geometry."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from vtuber_rigger.interfaces import MeshData
from vtuber_rigger.qa.geometry import check_geometry


# ─── Fixtures ───────────────────────────────────────────────────────────────

def make_mesh_data(verts: np.ndarray, faces: np.ndarray, **kwargs) -> MeshData:
    """Build a MeshData, inferring normals from faces if not given."""
    normals = kwargs.pop("normals", None)
    if normals is None and len(verts) > 0:
        tm = trimesh.Trimesh(vertices=verts, faces=faces)
        normals = tm.vertex_normals.astype(np.float32)
    return MeshData(vertices=verts, faces=faces, normals=normals, **kwargs)


def sphere_mesh(n: int = 3) -> MeshData:
    """Ico-sphere with n subdivisions, normalized to [-1,1]³."""
    sphere = trimesh.creation.icosphere(subdivisions=n)
    v = np.asarray(sphere.vertices, dtype=np.float32)
    center = v.min(axis=0) + (v.max(axis=0) - v.min(axis=0)) / 2
    scale = 2.0 / max((v.max(axis=0) - v.min(axis=0)).max(), 1e-10)
    v = (v - center) * scale
    f = np.asarray(sphere.faces, dtype=np.int32)
    return make_mesh_data(v, f)


def box_mesh(size: float = 2.0) -> MeshData:
    """A subdivided box (enough vertices to pass M1-4 >= 100), normalized."""
    tm = trimesh.creation.box(extents=[size, size, size])
    # Subdivide to get more vertices (3 subdivisions for >100 verts)
    tm = tm.subdivide()
    tm = tm.subdivide()
    tm = tm.subdivide()
    v = np.asarray(tm.vertices, dtype=np.float32)
    # Normalize to [-1,1]
    center = v.min(axis=0) + (v.max(axis=0) - v.min(axis=0)) / 2
    scale = 2.0 / max((v.max(axis=0) - v.min(axis=0)).max(), 1e-10)
    v = ((v - center) * scale).astype(np.float32)
    f = np.asarray(tm.faces, dtype=np.int32)
    return make_mesh_data(v, f)


def torus_mesh() -> MeshData:
    """A torus with enough vertices, normalized to [-1,1]."""
    tm = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_sections=32, minor_sections=16)
    v = np.asarray(tm.vertices, dtype=np.float32)
    # Normalize to [-1,1]
    center = v.min(axis=0) + (v.max(axis=0) - v.min(axis=0)) / 2
    scale = 2.0 / max((v.max(axis=0) - v.min(axis=0)).max(), 1e-10)
    v = ((v - center) * scale).astype(np.float32)
    f = np.asarray(tm.faces, dtype=np.int32)
    return make_mesh_data(v, f)


# ─── Valid mesh tests ───────────────────────────────────────────────────────

class TestValidGeometry:
    def test_sphere_is_valid(self):
        mesh = sphere_mesh()
        errors = check_geometry(mesh)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_box_is_valid(self):
        mesh = box_mesh()
        errors = check_geometry(mesh)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_torus_is_valid(self):
        mesh = torus_mesh()
        errors = check_geometry(mesh)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_sphere_with_uvs_valid(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        v = np.asarray(sphere.vertices, dtype=np.float32)
        f = np.asarray(sphere.faces, dtype=np.int32)
        uvs = np.zeros((len(v), 2), dtype=np.float32)
        uvs[:, 0] = (v[:, 0] + 1) / 2
        uvs[:, 1] = (v[:, 1] + 1) / 2
        mesh = make_mesh_data(v, f, uvs=uvs)
        errors = check_geometry(mesh)
        assert errors == [], f"expected no errors, got: {errors}"


# ─── Invalid mesh tests ──────────────────────────────────────────────────────

class TestInvalidGeometry:

    def test_nan_vertex(self):
        """A vertex with NaN should be rejected."""
        mesh = sphere_mesh()
        mesh.vertices[0, 0] = np.nan
        errors = check_geometry(mesh)
        assert any("NaN" in e for e in errors), f"expected NaN error, got: {errors}"

    def test_inf_vertex(self):
        """A vertex with Inf should be rejected."""
        mesh = sphere_mesh()
        mesh.vertices[0, 0] = np.inf
        errors = check_geometry(mesh)
        assert any("Inf" in e for e in errors), f"expected Inf error, got: {errors}"

    def test_too_many_faces(self):
        """Face count > 50 000 should be rejected."""
        sphere = trimesh.creation.icosphere(subdivisions=6)  # 81920 faces > 50000
        v = np.asarray(sphere.vertices, dtype=np.float32)
        f = np.asarray(sphere.faces, dtype=np.int32)
        mesh = make_mesh_data(v, f)
        errors = check_geometry(mesh)
        assert any("50" in e and "000" in e or "exceed" in e or "poly" in e.lower() or "face" in e.lower() for e in errors), \
            f"expected poly count error, got: {errors}"

    def test_degenerate_faces(self):
        """Two identical triangles share the same area ~0 — degenerate."""
        v = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=np.float32)
        # Three degenerate (zero-area) faces
        f = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int32)
        mesh = make_mesh_data(v, f)
        errors = check_geometry(mesh)
        assert any("degenerate" in e for e in errors), f"expected degenerate error, got: {errors}"

    def test_open_surface_not_watertight(self):
        """An open cylinder is not watertight."""
        tm = trimesh.creation.cylinder(radius=0.5, height=1.0, sections=16)
        # Remove the caps to make it open
        tm = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces[tm.faces[:, 1] != tm.faces[:, 0]])  # rough
        # Or just use a simple open mesh
        v = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [0.5, 0.5, 1.0],
        ], dtype=np.float32)
        f = np.array([[0, 1, 2], [1, 3, 2], [0, 2, 4], [1, 4, 3]], dtype=np.int32)
        mesh = make_mesh_data(v, f)
        errors = check_geometry(mesh)
        assert any("watertight" in e for e in errors), f"expected watertight error, got: {errors}"

    def test_vertex_count_too_low(self):
        """Mesh with < 100 vertices should be rejected."""
        v = np.random.randn(50, 3).astype(np.float32)
        f = np.array([[0, 1, 2] for _ in range(50)], dtype=np.int32)
        mesh = make_mesh_data(v, f)
        errors = check_geometry(mesh)
        assert any("100" in e for e in errors), f"expected vertex count error, got: {errors}"

    def test_nonfinite_uv(self):
        """Non-finite UV coordinates should be rejected."""
        mesh = sphere_mesh()
        mesh.uvs = np.zeros((mesh.vertex_count, 2), dtype=np.float32)
        mesh.uvs[0, 0] = np.nan
        errors = check_geometry(mesh)
        assert any("UV" in e or "NaN" in e for e in errors), \
            f"expected UV error, got: {errors}"

    def test_bounding_box_exceeds_limit(self):
        """Vertices outside [-1,1]³ should be rejected."""
        mesh = sphere_mesh()
        mesh.vertices[:, 0] *= 2.5  # scale beyond unit cube
        errors = check_geometry(mesh)
        assert any("bounding" in e or "[-1" in e for e in errors), \
            f"expected bounding box error, got: {errors}"
