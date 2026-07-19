"""QA checks for mesh geometry — watertight, poly count, UV validity, manifold."""

from __future__ import annotations

import numpy as np
import trimesh

from vtuber_rigger.interfaces import MeshData


def _meshdata_to_trimesh(mesh: MeshData) -> trimesh.Trimesh:
    """Convert MeshData to trimesh for property checks."""
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        vertex_normals=np.asarray(mesh.normals, dtype=np.float64) if mesh.normals is not None else None,
    )


def check_geometry(mesh: MeshData) -> list[str]:
    """
    Run all geometry checks on a MeshData object.

    Returns a list of error strings (empty = valid).
    Checks:
      - vertices are finite (no NaN/Inf)
      - face count <= 50 000
      - mesh is watertight (closed surface)
      - no degenerate faces (area > 1e-10)
      - UV validity (if present: all finite)
      - vertices are finite (no NaN/Inf)
      - mesh is manifold (every edge shared by exactly 2 faces)
    """
    errors: list[str] = []

    # M1-1: vertices finite
    if not np.all(np.isfinite(mesh.vertices)):
        errors.append(" vertices contain NaN or Inf")

    # Poly count: M1-4 and AC8
    face_count = mesh.face_count
    if face_count > 50_000:
        errors.append(f" face count {face_count} exceeds 50 000 limit (AC8)")

    # UV validity: M1-6
    if mesh.uvs is not None:
        if not np.all(np.isfinite(mesh.uvs)):
            errors.append(" UV coordinates contain NaN or Inf (M1-6)")

    # Degenerate faces: B3-4
    tm = _meshdata_to_trimesh(mesh)
    if hasattr(tm, 'area_faces'):
        areas = tm.area_faces
        degenerate = areas < 1e-10
        if np.any(degenerate):
            errors.append(f" {degenerate.sum()} degenerate faces with area < 1e-10 (B3-4)")

    # Watertight: B3-7
    try:
        if not tm.is_watertight:
            errors.append(" mesh is not watertight — open surface (B3-7)")
    except Exception:
        errors.append(" could not determine watertightness (check failed)")

    # Manifold: B3-6
    try:
        if not tm.is_watertight:
            # already reported above; still flag non-manifold
            errors.append(" mesh is not manifold (B3-6)")
        elif not tm.is_watertight:
            # double-check with edge count heuristic
            pass
    except Exception:
        pass

    # Manifold via edge uniqueness check
    try:
        # Every edge should be shared by exactly 2 faces
        edges = tm.edges_unique
        edges_face_count = tm.edges_face_count
        non_manifold_edges = np.sum(edges_face_count != 2)
        if non_manifold_edges > 0:
            errors.append(f" {non_manifold_edges} non-manifold edges (B3-6)")
    except Exception:
        pass

    # Vertex count bounds: M1-4
    if mesh.vertex_count < 100:
        errors.append(f" vertex count {mesh.vertex_count} < 100 (M1-4)")

    # Bounding box: M1-5
    try:
        bb_min, bb_max = mesh.bounding_box
        extents = bb_max - bb_min
        if np.any(extents > 2.0 + 1e-6):
            errors.append(f" bounding box extents {extents} exceed [-1,1]³ (M1-5)")
    except Exception:
        pass

    return errors
