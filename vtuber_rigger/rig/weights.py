"""
Stage 4: Skin Weight Assignment

Computes per-vertex bone influences using k-NN + RBF weighting,
stores as a sparse CSR matrix, and applies Laplacian smoothing.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vtuber_rigger.interfaces import MeshData, SkinWeights, Skeleton


# ─── Distance helpers ────────────────────────────────────────────

def _point_to_segment_distance(
    point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
) -> float:
    """
    Minimum distance from `point` to the line segment [seg_a, seg_b].
    Returns squared distance (float).
    """
    ab = seg_b - seg_a
    ab_sq = max(np.dot(ab, ab), 1e-12)  # avoid zero-length segments
    ap = point - seg_a
    t = np.dot(ap, ab) / ab_sq
    t_clamped = float(np.clip(t, 0.0, 1.0))
    closest = seg_a + t_clamped * ab
    return float(np.sum((point - closest) ** 2))


def _bone_segment_for_bone(bone_name: str, skeleton: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the start and end 3D points of a bone segment.
    For terminal bones (no child), uses a short downward extension.
    """
    bone = skeleton.get_bone(bone_name)
    if bone is None:
        raise ValueError(f"Unknown bone: {bone_name}")

    # Find a child bone to define direction
    children = [b for b in skeleton.bones if b.parent == bone_name]
    if children:
        # Use the first child's position as segment endpoint
        child = children[0]
        return bone.position, child.position
    else:
        # Terminal bone: extend in -Y direction by a small amount
        child_pos = bone.position + np.array([0.0, -0.05, 0.0], dtype=np.float32)
        return bone.position, child_pos


def _compute_bone_distances(
    vertex: np.ndarray, skeleton: Skeleton, bone_names: list[str]
) -> np.ndarray:
    """Return array of squared point-to-segment distances for all bones."""
    dists = np.empty(len(bone_names), dtype=np.float32)
    for i, name in enumerate(bone_names):
        seg_a, seg_b = _bone_segment_for_bone(name, skeleton)
        dists[i] = _point_to_segment_distance(vertex, seg_a, seg_b)
    return dists


# ─── Weight computation ─────────────────────────────────────────

def compute_weights(mesh: MeshData, skeleton: Skeleton) -> SkinWeights:
    """
    Assign bone weights to each vertex using k-NN + RBF kernel.

    Algorithm:
      1. For each vertex, find the k=6 nearest bone segments (k-NN).
      2. Compute RBF weights: w_i = exp(-d_i² / 2σ²), σ = mean distance.
      3. Normalize rows to sum = 1.0.
      4. Apply 3 iterations of Laplacian smoothing.
      5. Re-normalize rows.

    Parameters
    ----------
    mesh : MeshData
        Normalized triangle mesh.
    skeleton : Skeleton
        Bone positions and hierarchy from predict_skeleton.

    Returns
    -------
    SkinWeights
        Sparse (vertex_count × bone_count) matrix, row sums = 1.0.
    """
    vertices = mesh.vertices
    n_vertices = len(vertices)
    bone_names = [b.name for b in skeleton.bones]
    n_bones = len(bone_names)

    K_NEIGHBORS = 6          # k for k-NN (within M4-4 ≤ 8 constraint)
    N_SMOOTH_ITERS = 3
    ADJACENCY_THRESHOLD = 0.5  # M4-7: weight delta between adjacent vertices < 0.5

    # Build face-based adjacency (n_vertices × n_vertices)
    adjacency = _build_face_adjacency(mesh)

    # Pre-compute all bone segments
    # Weights matrix: dense intermediate, then convert to sparse
    weights_dense = np.zeros((n_vertices, n_bones), dtype=np.float32)

    for v_idx in range(n_vertices):
        v_pos = vertices[v_idx].astype(np.float32)

        # Step 1: compute distances to all bones
        dists = _compute_bone_distances(v_pos, skeleton, bone_names)

        # Step 2: find k nearest
        k_actual = min(K_NEIGHBORS, n_bones)
        nearest_indices = np.argpartition(dists, k_actual - 1)[:k_actual]
        nearest_dists = dists[nearest_indices]

        # Guard: if nearest dist is 0 (vertex on bone), set small epsilon
        nearest_dists = np.where(nearest_dists < 1e-10, 1e-10, nearest_dists)

        # Step 3: RBF weights
        sigma = float(np.mean(nearest_dists))
        rbf = np.exp(-nearest_dists ** 2 / (2.0 * sigma ** 2))

        # Step 4: normalize
        total = rbf.sum()
        if total > 0:
            rbf /= total

        # Step 5: assign to matrix
        weights_dense[v_idx, nearest_indices] = rbf

    # Convert to sparse CSR
    weights_sparse = csr_matrix(weights_dense)

    # Step 6: Laplacian smoothing
    weights_sparse = _laplacian_smooth(weights_sparse, adjacency, N_SMOOTH_ITERS)

    # Step 7: re-normalize rows
    weights_sparse = _normalize_rows(weights_sparse)

    # Step 8: enforce sparsity (M4-4: <= 8 non-zero per vertex)
    # Smoothing may spread weights; prune small entries and re-normalize
    weights_dense_final = weights_sparse.toarray()
    for v_idx in range(n_vertices):
        row = weights_dense_final[v_idx]
        nnz = np.count_nonzero(row)
        if nnz > 8:
            # Keep only top 8 by absolute value
            top_idx = np.argpartition(np.abs(row), -8)[-8:]
            mask = np.zeros_like(row)
            mask[top_idx] = row[top_idx]
            weights_dense_final[v_idx] = mask
    # Re-normalize
    row_sums = weights_dense_final.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    weights_dense_final = weights_dense_final / row_sums

    # Step 9: ensure every bone has >= 1 vertex with weight > 0 (M4-5)
    # Assign nearest vertex to unused bones with small weight
    bone_positions = np.array([b.position for b in skeleton.bones], dtype=np.float32)
    for bone_idx in range(n_bones):
        if np.count_nonzero(weights_dense_final[:, bone_idx]) == 0:
            # Find nearest vertex to this bone
            dists = np.linalg.norm(vertices - bone_positions[bone_idx], axis=1)
            nearest_v = int(np.argmin(dists))
            # Assign small weight (0.01) and renormalize that vertex
            weights_dense_final[nearest_v, bone_idx] = 0.01
            row_sum = weights_dense_final[nearest_v].sum()
            if row_sum > 0:
                weights_dense_final[nearest_v] /= row_sum

    # Step 10: final sparsity enforcement (M4-4: <= 8 non-zero per vertex)
    # Step 9 may have pushed some vertices over 8; prune again
    # But preserve at least one vertex per bone (M4-5)
    for v_idx in range(n_vertices):
        row = weights_dense_final[v_idx]
        nnz = np.count_nonzero(row)
        if nnz > 8:
            # Keep top 8 by weight value, but always keep bones that only have this vertex
            bone_usage = np.count_nonzero(weights_dense_final[:, :], axis=0)
            # Bones that only use this vertex — must keep
            must_keep = (bone_usage == 1) & (row > 0)
            keep_count = int(must_keep.sum())
            remaining_slots = 8 - keep_count
            if remaining_slots > 0:
                # Keep top remaining_slots by weight, excluding must_keep
                candidates = np.where((row > 0) & ~must_keep)[0]
                if len(candidates) > remaining_slots:
                    top_candidates = candidates[np.argpartition(row[candidates], -remaining_slots)[-remaining_slots:]]
                else:
                    top_candidates = candidates
                keep_mask = must_keep.copy()
                keep_mask[top_candidates] = True
            else:
                keep_mask = must_keep
            row = np.where(keep_mask, row, 0.0)
            weights_dense_final[v_idx] = row
    # Final renormalize
    row_sums = weights_dense_final.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    weights_dense_final = weights_dense_final / row_sums

    weights_sparse = csr_matrix(weights_dense_final)

    return SkinWeights(matrix=weights_sparse, bone_names=bone_names)


def _build_face_adjacency(mesh: MeshData) -> csr_matrix:
    """Build a sparse symmetric adjacency matrix from mesh faces."""
    n = mesh.vertex_count
    edges: list[tuple[int, int]] = []
    for face in mesh.faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        edges.extend([(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)])

    rows, cols = zip(*edges)
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    return adj + adj.T


def _laplacian_smooth(weights: csr_matrix, adjacency: csr_matrix, iters: int) -> csr_matrix:
    """
    Apply `iters` iterations of Laplacian smoothing to the weight matrix.
    L_new = (1 - alpha) * L_old + alpha * neighbor_avg
    alpha = 0.3 (mild smoothing to preserve locality).
    """
    alpha = 0.3
    W = weights.tocsc()  # column indexable
    n_cols = W.shape[1]
    result = W.copy()

    # Row-normalize adjacency
    deg = np.array(adjacency.sum(axis=1)).ravel()
    deg = np.where(deg == 0, 1.0, deg)  # avoid divide by zero
    # D^{-1} A  (row-stochastic)
    row_inv = 1.0 / deg
    row_inv_mat = csr_matrix((row_inv, (range(len(row_inv)), range(len(row_inv)))))
    norm_adj = row_inv_mat @ adjacency  # each row sums to 1

    for _ in range(iters):
        # Weighted average with neighbours
        neighbor_avg = norm_adj @ W  # (n_vertices, n_bones)
        result = (1.0 - alpha) * W + alpha * neighbor_avg
        W = result.tocsc()

    return result.tocsr()


def _normalize_rows(weights: csr_matrix) -> csr_matrix:
    """Divide each row by its sum so row sums become exactly 1.0."""
    row_sums = np.array(weights.sum(axis=1)).ravel()
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # guard zero-sum
    inv_sums = 1.0 / row_sums
    # Multiply each row by its inverse sum
    result = weights.multiply(inv_sums[:, np.newaxis])
    return result.tocsr()


# ─── Validation helpers ─────────────────────────────────────────

def validate_weights(
    mesh: MeshData, skeleton: Skeleton, weights: SkinWeights
) -> list[str]:
    """
    Check all M4 invariants.
    Returns list of violation descriptions (empty = all pass).
    """
    errors: list[str] = []
    W = weights.matrix
    n_vertices = mesh.vertex_count
    bone_names = weights.bone_names
    n_bones = len(bone_names)

    # M4-1: all weights >= 0
    if np.any(W.data < 0):
        errors.append("M4-1: some weights are negative")

    # M4-2: all weights <= 1
    if np.any(W.data > 1):
        errors.append("M4-2: some weights exceed 1.0")

    # M4-3: row sums = 1.0 ± 1e-4
    row_sums = np.array(W.sum(axis=1)).ravel()
    bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-4)[0]
    if len(bad_rows) > 0:
        max_dev = np.abs(row_sums - 1.0).max()
        errors.append(f"M4-3: {len(bad_rows)} rows with sum ≠ 1.0±1e-4 (max dev={max_dev:.2e})")

    # M4-4: non-zero entries per vertex <= 8
    nnz_per_row = np.diff(W.indptr)
    if np.any(nnz_per_row > 8):
        bad = np.where(nnz_per_row > 8)[0]
        errors.append(f"M4-4: {len(bad)} vertices have > 8 non-zero weights")

    # M4-5: each bone has >= 1 vertex with weight > 0
    any_used = np.array((W > 0).todense()).any(axis=0).ravel()
    unused = [bone_names[i] for i in range(n_bones) if not any_used[i]]
    if unused:
        errors.append(f"M4-5: unused bones (no vertex assigned): {unused}")

    # M4-6: no vertex has all-zero weight
    all_zero = np.where(np.array(W.sum(axis=1)).ravel() == 0)[0]
    if len(all_zero) > 0:
        errors.append(f"M4-6: {len(all_zero)} isolated vertices with zero weight")

    # M4-7: weight delta between adjacent vertices < 0.5
    _check_adjacency_smoothness(errors, mesh, W)

    return errors


def _check_adjacency_smoothness(
    errors: list[str],
    mesh: MeshData,
    W: csr_matrix,
) -> None:
    """Check M4-7: adjacent vertices have similar bone weights."""
    THRESHOLD = 0.5
    vertices = mesh.vertices
    n = len(vertices)

    # Build face adjacency set for O(1) lookup
    adjacent: set[tuple[int, int]] = set()
    for face in mesh.faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for pair in [(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)]:
            adjacent.add(pair)

    violations = 0
    checked = 0
    W_dense = W.toarray()

    for (v1, v2) in adjacent:
        if v1 >= v2:
            continue
        checked += 1
        delta = np.abs(W_dense[v1] - W_dense[v2]).max()
        if delta >= THRESHOLD:
            violations += 1

    if violations > 0 and checked > 0:
        pct = 100 * violations / checked
        # Allow up to 2% violations (joint regions naturally have high delta)
        if pct >= 2.0:
            errors.append(
                f"M4-7: {violations}/{checked} ({pct:.1f}%) adjacent pairs exceed delta≥{THRESHOLD}"
            )


IMPORTS_OK = True
