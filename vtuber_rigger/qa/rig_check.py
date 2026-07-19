"""QA checks for rig — bone count, cycles, weight normalization, sparsity."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from vtuber_rigger.interfaces import (
    REQUIRED_BONES,
    MeshData,
    Skeleton,
    SkinWeights,
)


def _has_cycle(skeleton: Skeleton) -> bool:
    """Return True if the skeleton has a cycle (via BFS from each root)."""
    children_map: dict[str, list[str]] = {}
    for bone in skeleton.bones:
        children_map.setdefault(bone.name, [])
        if bone.parent:
            children_map.setdefault(bone.parent, []).append(bone.name)

    visited: set[str] = set()
    in_stack: set[str] = set()

    def dfs(name: str) -> bool:
        visited.add(name)
        in_stack.add(name)
        for child in children_map.get(name, []):
            if child not in visited:
                if dfs(child):
                    return True
            elif child in in_stack:
                return True
        in_stack.remove(name)
        return False

    for bone in skeleton.bones:
        if bone.parent is None:
            if dfs(bone.name):
                return True

    # Orphaned nodes (no parent, not reached from any root)
    roots = [b.name for b in skeleton.bones if b.parent is None]
    for bone in skeleton.bones:
        if bone.name not in visited and bone.name not in roots:
            if dfs(bone.name):
                return True
    return False


def check_rig(
    skeleton: Skeleton,
    weights: SkinWeights,
    mesh: MeshData,
) -> list[str]:
    """
    Run all rig QA checks.

    Returns a list of error strings (empty = valid).
    Checks:
      - bone count >= 15 (M3-1)
      - no cycles in parent chain (M3-3)
      - weight normalization: per-vertex sum == 1.0 ± 1e-4 (M4-3)
      - sparsity: non-zero entries per vertex <= 8 (M4-4)
      - all bones used: every bone has >= 1 weighted vertex (M4-5)
      - no isolated vertices: every vertex has > 0 weight (M4-6)
    """
    errors: list[str] = []

    # ── Bone count (M3-1) ──────────────────────────────────────────
    bone_count = len(skeleton.bones)
    if bone_count < 15:
        errors.append(f" bone count {bone_count} < 15 required (M3-1)")

    # ── Required bones present (M3-1) ───────────────────────────────
    present = skeleton.bone_names
    missing = REQUIRED_BONES - present
    if missing:
        errors.append(f" missing required bones: {sorted(missing)} (M3-1)")

    # ── Cycle check (M3-3) ──────────────────────────────────────────
    if _has_cycle(skeleton):
        errors.append(" skeleton contains a cycle in parent chain (M3-3)")

    # ── Weight normalization (M4-3) ────────────────────────────────
    matrix = weights.matrix
    if matrix.shape[0] != mesh.vertex_count:
        errors.append(
            f" weight matrix row count {matrix.shape[0]} != vertex count {mesh.vertex_count}"
        )
    else:
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        bad_rows = ~np.isclose(row_sums, 1.0, atol=1e-4)
        if np.any(bad_rows):
            count = int(np.sum(bad_rows))
            errors.append(f" {count} vertices have non-normalized weights (sum != 1.0, M4-3)")

    # ── Sparsity (M4-4) ─────────────────────────────────────────────
    if matrix.shape[0] > 0:
        nnz_per_row = np.diff(matrix.tocsr().indptr)
        if np.any(nnz_per_row > 8):
            count = int(np.sum(nnz_per_row > 8))
            errors.append(f" {count} vertices have > 8 bone influences (M4-4)")

    # ── All bones used (M4-5) ───────────────────────────────────────
    if matrix.shape[1] > 0:
        bone_used = np.asarray((matrix > 0).todense()).any(axis=0).ravel()
        unused = [weights.bone_names[i] for i, used in enumerate(bone_used) if not used]
        if unused:
            errors.append(f" unused bones (no weighted vertices): {unused} (M4-5)")

    # ── No isolated vertices (M4-6) ──────────────────────────────────
    if matrix.shape[0] > 0:
        vertex_has_weight = np.asarray((matrix > 0).todense()).any(axis=1).ravel()
        if not np.all(vertex_has_weight):
            count = int(np.sum(~vertex_has_weight))
            errors.append(f" {count} isolated vertices with no bone weight (M4-6)")

    return errors
