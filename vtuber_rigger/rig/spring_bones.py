"""Stage 6: Spring Bone chain detection.

Detects elongated thin cross-section edge chains (hair/cloth candidates)
and maps them to VRM 1.0 spring bone chains with auto-placed colliders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.spatial

from vtuber_rigger.interfaces import (
    Collider,
    MeshData,
    Skeleton,
    SpringChain,
    SpringNode,
)


# ─── Parameters ─────────────────────────────────────────────────

# Minimum chain length (edges) to qualify as a spring chain
_MIN_CHAIN_EDGES = 2   # ≥2 nodes required by M6-1
_MIN_CHAIN_LENGTH = 0.15  # normalised world units

# Cross-section thinness threshold: mean edge length vs perpendicular radius
_THINNESS_RATIO = 0.35

# Default physics parameters (VRM spec ranges)
_DEFAULT_STIFFNESS = 0.6
_DEFAULT_DRAG = 0.4
_DEFAULT_HIT_RADIUS = 0.02
_DEFAULT_GRAVITY_POWER = 0.5
_DEFAULT_GRAVITY_DIR = np.array([0.0, -1.0, 0.0], dtype=np.float32)

# Default collider radii (in normalised mesh units)
_COLLIDER_RADII = {
    "head": 0.08,
    "shoulder": 0.05,
    "waist": 0.10,
}


# ─── Helpers ────────────────────────────────────────────────────

def _build_edge_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[int, set[int]]:
    """Build adjacency dict: vertex_index → set of neighbouring vertex indices."""
    adj: dict[int, set[int]] = {i: set() for i in range(len(vertices))}
    for f in faces:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _chains_from_adjacency(
    adj: dict[int, set[int]],
    vertices: np.ndarray,
    min_edges: int,
    min_length: float,
) -> list[list[int]]:
    """Find all simple paths (chains) in the mesh edge graph.

    Returns a list of chains, each chain is a list of vertex indices.
    Chains are maximal linear paths (degree ≤ 2 at interior nodes).
    """
    degree = {v: len(neibs) for v, neibs in adj.items()}

    # Find all degree-1 or degree-2 endpoints
    endpoints = [v for v, d in degree.items() if d == 1]
    interior = [v for v, d in degree.items() if d == 2]

    # Build a forest of linear chains by walking from each unvisited endpoint
    visited = set()
    chains: list[list[int]] = []

    for start in endpoints:
        if start in visited:
            continue
        chain: list[int] = [start]
        visited.add(start)
        prev = start

        while True:
            neighbors = [n for n in adj[prev] if n not in visited]
            if not neighbors:
                break
            nxt = neighbors[0]
            chain.append(nxt)
            visited.add(nxt)
            prev = nxt

        if len(chain) - 1 >= min_edges:
            # Compute total path length
            path_len = sum(
                np.linalg.norm(vertices[chain[i]] - vertices[chain[i + 1]])
                for i in range(len(chain) - 1)
            )
            if path_len >= min_length:
                chains.append(chain)

    # Also handle isolated loops (degree 0 or closed loops not reachable from endpoints)
    # — skip closed loops for spring bones (no root)
    return chains


def _thin_cross_section(
    chain: list[int],
    vertices: np.ndarray,
    threshold: float,
) -> bool:
    """Return True if the chain has a thin cross-section (hair-like).

    Checks: for each interior edge, compute the mean distance of nearby
    vertices from the edge axis. If below ``threshold * edge_length``,
    the cross-section is considered thin.
    """
    if len(chain) < 3:
        return True  # Short chains are assumed thin

    kdtree = scipy.spatial.cKDTree(vertices)

    for i in range(len(chain) - 1):
        a = vertices[chain[i]]
        b = vertices[chain[i + 1]]
        edge_dir = b - a
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-8:
            continue
        edge_dir /= edge_len

        # Points near this edge midpoint
        mid = (a + b) * 0.5
        radius = edge_len * 1.5
        near_idx = kdtree.query_ball_point(mid, r=radius)
        if not near_idx:
            continue

        # Distance of each point from the edge line
        max_dist = 0.0
        for idx in near_idx:
            if idx in (chain[i], chain[i + 1]):
                continue
            pt = vertices[idx]
            proj = np.dot(pt - a, edge_dir)
            closest = a + proj * edge_dir
            dist = np.linalg.norm(pt - closest)
            max_dist = max(max_dist, dist)

        # If perpendicular spread is small relative to edge length → thin
        if max_dist > threshold * edge_len:
            return False

    return True


def _map_chain_to_bone(
    chain: list[int],
    vertices: np.ndarray,
    skeleton: Skeleton,
) -> tuple[list[SpringNode], str]:
    """Map each chain node to the nearest existing skeleton bone.

    Returns (nodes, root_bone_name).
    """
    bone_positions = {b.name: b.position for b in skeleton.bones}
    nodes: list[SpringNode] = []

    for vi in chain:
        pos = vertices[vi]
        best_bone = min(
            bone_positions,
            key=lambda bn: np.linalg.norm(pos - bone_positions[bn])
        )
        nodes.append(SpringNode(bone_name=best_bone, position=pos.copy()))

    # Root bone: nearest bone to the first node's parent in hierarchy
    root_bone = nodes[0].bone_name
    return nodes, root_bone


def _auto_colliders(
    skeleton: Skeleton,
    mesh_bb_min: np.ndarray,
    mesh_bb_max: np.ndarray,
) -> list[Collider]:
    """Auto-place sphere colliders on head, shoulders, and waist."""
    colliders: list[Collider] = []
    center = (mesh_bb_min + mesh_bb_max) * 0.5

    def get_bone(name: str) -> Optional[np.ndarray]:
        b = skeleton.get_bone(name)
        return b.position.copy() if b else None

    # Head sphere
    head_pos = get_bone("head")
    if head_pos is not None:
        colliders.append(Collider(center=head_pos, radius=_COLLIDER_RADII["head"]))

    # Shoulder spheres (approximate as midpoint of upper arm bones)
    l_shoulder = get_bone("leftUpperArm")
    r_shoulder = get_bone("rightUpperArm")
    if l_shoulder is not None:
        colliders.append(Collider(center=l_shoulder, radius=_COLLIDER_RADII["shoulder"]))
    if r_shoulder is not None:
        colliders.append(Collider(center=r_shoulder, radius=_COLLIDER_RADII["shoulder"]))

    # Waist / hips sphere
    hips_pos = get_bone("hips")
    if hips_pos is not None:
        colliders.append(Collider(center=hips_pos, radius=_COLLIDER_RADII["waist"]))
    else:
        # Fallback: centre of bounding box at bottom quarter
        fallback = center.copy()
        fallback[1] = mesh_bb_min[1] + (mesh_bb_max[1] - mesh_bb_min[1]) * 0.15
        colliders.append(Collider(center=fallback, radius=_COLLIDER_RADII["waist"]))

    return colliders


# ─── Public API ─────────────────────────────────────────────────

def detect_spring_bones(
    mesh: MeshData,
    skeleton: Skeleton,
    labels: np.ndarray | None = None,
    *,
    stiffness: float = _DEFAULT_STIFFNESS,
    drag_force: float = _DEFAULT_DRAG,
    hit_radius: float = _DEFAULT_HIT_RADIUS,
    gravity_power: float = _DEFAULT_GRAVITY_POWER,
    gravity_dir: np.ndarray = _DEFAULT_GRAVITY_DIR,
) -> tuple[list[SpringChain], list[Collider]]:
    """Detect spring bone chains and colliders for the given mesh.

    Parameters
    ----------
    mesh : MeshData
        Normalized triangle mesh.
    skeleton : Skeleton
        VRM 1.0 bone hierarchy.
    labels : np.ndarray, optional
        Per-vertex labels from segmentation. If provided, chains not
        connected to ``"hair"`` or ``"clothing"`` labels are filtered out.
    stiffness : float
        Spring stiffness in [0.0, 1.0].
    drag_force : float
        Drag force in [0.0, 1.0].
    hit_radius : float
        Hit radius in [0.0, 1.0].
    gravity_power : float
        Gravity power in [0.0, 10.0].
    gravity_dir : np.ndarray
        Gravity direction unit vector.

    Returns
    -------
    tuple[list[SpringChain], list[Collider]]
        Detected spring chains and auto-placed sphere colliders.
    """
    if mesh.vertex_count == 0:
        return [], []

    # Build edge adjacency graph
    adj = _build_edge_graph(mesh.vertices, mesh.faces)

    # Extract all linear chains
    raw_chains = _chains_from_adjacency(
        adj, mesh.vertices,
        min_edges=_MIN_CHAIN_EDGES,
        min_length=_MIN_CHAIN_LENGTH,
    )

    chains: list[SpringChain] = []

    for chain in raw_chains:
        # Filter by label if available (hair / clothing only)
        if labels is not None:
            chain_labels = labels[chain]
            valid = any(l in ("hair", "clothing") for l in chain_labels)
            if not valid:
                continue

        # Filter by thin cross-section
        if not _thin_cross_section(chain, mesh.vertices, threshold=_THINNESS_RATIO):
            continue

        nodes, root_bone = _map_chain_to_bone(chain, mesh.vertices, skeleton)

        chains.append(SpringChain(
            nodes=nodes,
            root_bone=root_bone,
            stiffness=stiffness,
            dragForce=drag_force,
            hitRadius=hit_radius,
            gravityPower=gravity_power,
            gravityDir=gravity_dir.astype(np.float32),
        ))

    # Auto-place colliders
    bb_min, bb_max = mesh.bounding_box
    colliders = _auto_colliders(skeleton, bb_min, bb_max)

    return chains, colliders
