"""Stage 5: Expression (blend shape) generation.

Template-based approach (Approach 3) — precompute 17 displacement maps
on a generic face and apply them to the target mesh.
"""

from __future__ import annotations

import numpy as np

from vtuber_rigger.interfaces import (
    Expression,
    MeshData,
    MorphTarget,
    Skeleton,
    VRM_STANDARD_EXPRESSIONS,
)


# ─── Parameter ──────────────────────────────────────────────────

# Fraction of top Y vertices considered "head region" for expression morphing
_HEAD_FRACTION = 0.20


# ─── Helpers ────────────────────────────────────────────────────

def _head_mask(mesh: MeshData) -> np.ndarray:
    """Return bool mask for vertices in the top HEAD_FRACTION by Y (head region)."""
    if mesh.vertex_count == 0:
        return np.array([], dtype=bool)
    max_y = mesh.vertices[:, 1].max()
    threshold = max_y - (max_y - mesh.vertices[:, 1].min()) * _HEAD_FRACTION
    return mesh.vertices[:, 1] >= threshold


def _displacement_for_expression(
    base: np.ndarray,
    expr: str,
    head_mask: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex displacement for one expression.

    Parameters
    ----------
    base : (N, 3)
        Base vertex positions.
    expr : str
        Expression name.
    head_mask : (N,) bool
        Which vertices are in the head region.

    Returns
    -------
    (N, 3)
        Absolute vertex positions after this expression is applied at weight=1.
    """
    base = np.asarray(base, dtype=np.float32)
    delta = np.zeros_like(base)

    if expr == "happy":
        # Corners of mouth curve upward
        mask = head_mask & (np.abs(base[:, 0]) > 0.2) & (base[:, 1] < 0)
        delta[mask, 1] = 0.08
        # cheek raise
        mask = head_mask & (np.abs(base[:, 0]) > 0.25) & (base[:, 1] > 0)
        delta[mask, 1] = 0.04

    elif expr == "angry":
        # Brow lower, inner corners down
        brow = head_mask & (base[:, 1] > 0.15) & (np.abs(base[:, 0]) < 0.3)
        delta[brow, 1] = -0.06
        # slight mouth frown
        mouth = head_mask & (base[:, 1] < -0.1)
        delta[mouth, 1] = -0.04

    elif expr == "sad":
        # Inner brow raiser, mouth corners down
        inner_brow = head_mask & (np.abs(base[:, 0]) < 0.1) & (base[:, 1] > 0.1)
        delta[inner_brow, 1] = 0.05
        mouth = head_mask & (base[:, 1] < -0.1) & (np.abs(base[:, 0]) < 0.2)
        delta[mouth, 1] = -0.05

    elif expr == "relaxed":
        # Slight mouth open, brow neutral
        mouth = head_mask & (base[:, 1] < -0.05)
        delta[mouth, 1] = 0.03
        delta[mouth, 2] = 0.02

    elif expr == "surprised":
        # Wide eyes, open mouth, raised brows
        eyes = head_mask & (np.abs(base[:, 0]) < 0.25) & (base[:, 1] > 0.0)
        delta[eyes, 1] = 0.05
        delta[eyes, 2] = 0.03
        mouth = head_mask & (base[:, 1] < -0.15)
        delta[mouth, 1] = 0.08

    elif expr == "lookUp":
        # Shift eye vertices upward in Y
        eye = head_mask & (np.abs(base[:, 0]) < 0.3) & (base[:, 1] > 0.0) & (base[:, 1] < 0.25)
        delta[eye, 1] = 0.06

    elif expr == "lookDown":
        eye = head_mask & (np.abs(base[:, 0]) < 0.3) & (base[:, 1] > -0.1) & (base[:, 1] < 0.2)
        delta[eye, 1] = -0.05

    elif expr == "lookLeft":
        eye = head_mask & (base[:, 0] > -0.05) & (base[:, 0] < 0.3) & (base[:, 1] > 0.0) & (base[:, 1] < 0.25)
        delta[eye, 0] = -0.06

    elif expr == "lookRight":
        eye = head_mask & (base[:, 0] > -0.3) & (base[:, 0] < 0.05) & (base[:, 1] > 0.0) & (base[:, 1] < 0.25)
        delta[eye, 0] = 0.06

    elif expr == "blink":
        # Both upper eyelids drop
        lids = head_mask & (np.abs(base[:, 0]) < 0.3) & (base[:, 1] > 0.0) & (base[:, 1] < 0.18)
        delta[lids, 1] = -0.06

    elif expr == "blinkLeft":
        lids = head_mask & (base[:, 0] < 0.05) & (base[:, 0] > -0.3) & (base[:, 1] > 0.0) & (base[:, 1] < 0.18)
        delta[lids, 1] = -0.07

    elif expr == "blinkRight":
        lids = head_mask & (base[:, 0] > -0.05) & (base[:, 0] < 0.3) & (base[:, 1] > 0.0) & (base[:, 1] < 0.18)
        delta[lids, 1] = -0.07

    elif expr == "aa":
        # Jaw drops, mouth opens wide
        jaw = head_mask & (base[:, 1] < -0.05)
        delta[jaw, 1] = 0.12
        delta[jaw, 2] = 0.03

    elif expr == "ih":
        # Mouth corners slightly forward, jaw slightly down
        mouth = head_mask & (base[:, 1] < -0.05)
        delta[mouth, 1] = 0.04
        delta[mouth, 0] = 0.04 * np.sign(base[mouth, 0])

    elif expr == "ou":
        # Lips purse: protrude and narrow
        lips = head_mask & (base[:, 1] < -0.05) & (np.abs(base[:, 0]) < 0.2)
        delta[lips, 2] = 0.06
        delta[lips, 0] = -0.03 * np.sign(base[lips, 0])

    elif expr == "ee":
        # Slight smile, teeth show
        mouth = head_mask & (base[:, 1] < -0.05)
        delta[mouth, 1] = 0.03
        delta[mouth, 0] = 0.05 * np.sign(base[mouth, 0])

    elif expr == "oh":
        # Mouth rounds: expand horizontally, drop jaw
        mouth = head_mask & (base[:, 1] < -0.05) & (np.abs(base[:, 0]) < 0.2)
        delta[mouth, 1] = 0.08
        delta[mouth, 0] = 0.04 * np.sign(base[mouth, 0])

    return base + delta


# ─── Public API ─────────────────────────────────────────────────

def create_expressions(mesh: MeshData, skeleton: Skeleton) -> list[Expression]:
    """Generate all 17 VRM 1.0 standard expressions for the given mesh.

    Parameters
    ----------
    mesh : MeshData
        Normalized target mesh (vertex_count = N).
    skeleton : Skeleton
        VRM 1.0 bone hierarchy (unused in this template approach).

    Returns
    -------
    list[Expression]
        Ordered list of 17 expressions; each morph_target has exactly
        ``mesh.vertex_count`` vertices.
    """
    if mesh.vertex_count == 0:
        raise ValueError("Cannot create expressions for mesh with 0 vertices")

    base = mesh.vertices
    head_mask = _head_mask(mesh)
    expressions: list[Expression] = []

    for morph_target_index, expr_name in enumerate(sorted(VRM_STANDARD_EXPRESSIONS)):
        morph_vertices = _displacement_for_expression(base, expr_name, head_mask)

        morph = MorphTarget(
            name=expr_name,
            vertices=morph_vertices,
            vertex_count=mesh.vertex_count,
        )

        expressions.append(Expression(
            name=expr_name,
            morph_target=morph,
            morph_target_index=morph_target_index,
            preset_weight=1.0,
        ))

    return expressions
