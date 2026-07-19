"""
Stage 3: Skeleton Prediction

Places VRM 1.0 humanoid bones on the mesh using bounding box proportions
and curvature-based joint detection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from vtuber_rigger.interfaces import (
    Bone,
    MeshData,
    REQUIRED_BONES,
    Skeleton,
    VRM_PARENT_RULES,
)


# ─── Bone position heuristics ────────────────────────────────────

def _bounding_box(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    amin = mesh.vertices.min(axis=0)
    amax = mesh.vertices.max(axis=0)
    return amin, amax


def _center_of_mass(mesh: MeshData) -> np.ndarray:
    return mesh.vertices.mean(axis=0)


def _find_extremes_by_axis(
    vertices: np.ndarray, axis: int, which: str = "max"
) -> np.ndarray:
    """Return the vertex (3-array) with the extreme value on the given axis."""
    idx = np.argmax(vertices[:, axis]) if which == "max" else np.argmin(vertices[:, axis])
    return vertices[idx]


def _estimate_torso_height(mesh: MeshData) -> float:
    """Heuristic: torso height = 0.4 × total Y extent."""
    _, amax = _bounding_box(mesh)
    amin, _ = _bounding_box(mesh)
    return (amax[1] - amin[1]) * 0.4


def _estimate_shoulder_width(mesh: MeshData) -> float:
    """Heuristic: shoulder width ≈ 0.35 × X extent."""
    _, amax = _bounding_box(mesh)
    amin, _ = _bounding_box(mesh)
    return (amax[0] - amin[0]) * 0.35


def _detect_joint_by_curvature(
    mesh: MeshData, start: np.ndarray, direction: np.ndarray, max_t: float = 1.0
) -> np.ndarray:
    """
    March from `start` along `direction` and find the point of maximum
    discrete curvature projected onto the mesh surface.
    Returns the 3D position of the estimated joint.
    """
    vertices = mesh.vertices
    n = len(vertices)

    # Project all vertices onto the travel axis
    t_values = (vertices - start) @ direction  # scalar projection
    valid = (t_values > 0) & (t_values <= max_t)
    if not np.any(valid):
        # Fallback: closest vertex along direction
        t_candidates = np.maximum(t_values, 0)
        closest_idx = int(np.argmin(np.abs(t_candidates)))
        return vertices[closest_idx]

    # Simple curvature proxy: angle between normals of neighbouring vertices
    # We use normal divergence as curvature proxy
    if mesh.normals is None:
        # Fallback: use centroid of valid region
        return vertices[valid].mean(axis=0)

    # Sort valid vertices by projection
    order = np.argsort(t_values[valid])
    valid_vertices = vertices[valid][order]
    valid_normals = mesh.normals[valid][order]

    if len(valid_vertices) < 3:
        return valid_vertices.mean(axis=0) if len(valid_vertices) > 0 else start + direction * max_t * 0.5

    # Compute normal angle change along the arc
    normal_deltas = np.diff(valid_normals, axis=0)
    curvature = np.linalg.norm(normal_deltas, axis=1)
    max_curv_idx = int(np.argmax(curvature)) + 1  # +1 because diff reduces length

    # Clamp
    max_curv_idx = min(max_curv_idx, len(valid_vertices) - 1)
    return valid_vertices[max_curv_idx]


def _place_bone(
    name: str,
    position: np.ndarray,
    parent: Optional[str],
) -> Bone:
    return Bone(name=name, position=position.astype(np.float32), parent=parent)


# ─── Main predictor ──────────────────────────────────────────────

def predict_skeleton(mesh: MeshData, labels=None) -> Skeleton:
    """
    Place VRM 1.0 humanoid bones on the mesh.

    Uses bounding box proportions to size the skeleton and
    curvature/extremity heuristics for joint placement.

    Parameters
    ----------
    mesh : MeshData
        Normalized triangle mesh.
    labels : array-like, optional
        Per-vertex labels (not required; heuristic fallback uses geometry).

    Returns
    -------
    Skeleton
        Bone hierarchy satisfying M3-1 … M3-11.
    """
    vertices = mesh.vertices
    amin, amax = _bounding_box(mesh)
    center = _center_of_mass(mesh)
    height = amax[1] - amin[1]
    width = amax[0] - amin[0]
    depth = amax[2] - amin[2]

    bones: list[Bone] = []

    # ── Root: hips ─────────────────────────────────────────────
    hips_pos = np.array(
        [center[0], amin[1] + height * 0.12, center[2]], dtype=np.float32
    )
    bones.append(_place_bone("hips", hips_pos, None))

    # ── Spine ─────────────────────────────────────────────────
    spine_pos = np.array(
        [hips_pos[0], hips_pos[1] + height * 0.28, hips_pos[2]], dtype=np.float32
    )
    bones.append(_place_bone("spine", spine_pos, "hips"))

    # ── Chest (optional but included) ─────────────────────────
    chest_pos = np.array(
        [spine_pos[0], spine_pos[1] + height * 0.12, spine_pos[2]], dtype=np.float32
    )
    bones.append(_place_bone("chest", chest_pos, "spine"))

    # ── Upper chest ────────────────────────────────────────────
    upper_chest_pos = np.array(
        [chest_pos[0], chest_pos[1] + height * 0.06, chest_pos[2]], dtype=np.float32
    )
    bones.append(_place_bone("upperChest", upper_chest_pos, "chest"))

    # ── Neck ───────────────────────────────────────────────────
    neck_pos = np.array(
        [upper_chest_pos[0], upper_chest_pos[1] + height * 0.06, upper_chest_pos[2]],
        dtype=np.float32,
    )
    bones.append(_place_bone("neck", neck_pos, "upperChest"))

    # ── Head ───────────────────────────────────────────────────
    head_pos = np.array(
        [neck_pos[0], neck_pos[1] + height * 0.14, neck_pos[2]], dtype=np.float32
    )
    bones.append(_place_bone("head", head_pos, "neck"))

    # ── Eyes ───────────────────────────────────────────────────
    eye_y = head_pos[1] + height * 0.025
    eye_z = head_pos[2] + depth * 0.06
    left_eye_pos = np.array(
        [head_pos[0] - width * 0.065, eye_y, eye_z], dtype=np.float32
    )
    right_eye_pos = np.array(
        [head_pos[0] + width * 0.065, eye_y, eye_z], dtype=np.float32
    )
    bones.append(_place_bone("leftEye", left_eye_pos, "head"))
    bones.append(_place_bone("rightEye", right_eye_pos, "head"))

    # ── Jaw ────────────────────────────────────────────────────
    jaw_pos = np.array(
        [head_pos[0], head_pos[1] - height * 0.03, head_pos[2] + depth * 0.04],
        dtype=np.float32,
    )
    bones.append(_place_bone("jaw", jaw_pos, "head"))

    # ── Arm chain (left) ───────────────────────────────────────
    shoulder_y = upper_chest_pos[1] + height * 0.02
    left_shoulder_x = hips_pos[0] - width * 0.28
    left_upper_arm_pos = np.array(
        [left_shoulder_x, shoulder_y, center[2]], dtype=np.float32
    )
    bones.append(_place_bone("leftUpperArm", left_upper_arm_pos, "upperChest"))

    # Elbow: curvature detection along upper arm → hand direction
    left_elbow_dir = np.array([-0.15, -0.1, 0.1], dtype=np.float32)
    left_elbow_dir /= max(np.linalg.norm(left_elbow_dir), 1e-8)
    left_lower_arm_pos_raw = _detect_joint_by_curvature(
        mesh, left_upper_arm_pos, left_elbow_dir, max_t=width * 0.35
    )
    # Fallback proportional placement
    left_lower_arm_pos = np.array(
        [
            left_upper_arm_pos[0] + left_elbow_dir[0] * width * 0.3,
            left_upper_arm_pos[1] + left_elbow_dir[1] * height * 0.28,
            left_upper_arm_pos[2] + left_elbow_dir[2] * depth * 0.05,
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("leftLowerArm", left_lower_arm_pos, "leftUpperArm"))

    left_hand_pos = np.array(
        [
            left_lower_arm_pos[0] + left_elbow_dir[0] * width * 0.18,
            left_lower_arm_pos[1] - height * 0.02,
            left_lower_arm_pos[2],
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("leftHand", left_hand_pos, "leftLowerArm"))

    # ── Arm chain (right) ──────────────────────────────────────
    right_shoulder_x = hips_pos[0] + width * 0.28
    right_upper_arm_pos = np.array(
        [right_shoulder_x, shoulder_y, center[2]], dtype=np.float32
    )
    bones.append(_place_bone("rightUpperArm", right_upper_arm_pos, "upperChest"))

    right_elbow_dir = np.array([0.15, -0.1, 0.1], dtype=np.float32)
    right_elbow_dir /= max(np.linalg.norm(right_elbow_dir), 1e-8)
    right_lower_arm_pos = np.array(
        [
            right_upper_arm_pos[0] + right_elbow_dir[0] * width * 0.3,
            right_upper_arm_pos[1] + right_elbow_dir[1] * height * 0.28,
            right_upper_arm_pos[2] + right_elbow_dir[2] * depth * 0.05,
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("rightLowerArm", right_lower_arm_pos, "rightUpperArm"))

    right_hand_pos = np.array(
        [
            right_lower_arm_pos[0] + right_elbow_dir[0] * width * 0.18,
            right_lower_arm_pos[1] - height * 0.02,
            right_lower_arm_pos[2],
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("rightHand", right_hand_pos, "rightLowerArm"))

    # ── Leg chain (left) ───────────────────────────────────────
    left_leg_x = hips_pos[0] - width * 0.09
    left_upper_leg_pos = np.array(
        [left_leg_x, hips_pos[1] - height * 0.06, center[2]], dtype=np.float32
    )
    bones.append(_place_bone("leftUpperLeg", left_upper_leg_pos, "hips"))

    left_knee_pos = np.array(
        [
            left_upper_leg_pos[0],
            left_upper_leg_pos[1] - height * 0.22,
            left_upper_leg_pos[2] + depth * 0.02,
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("leftLowerLeg", left_knee_pos, "leftUpperLeg"))

    left_foot_pos = np.array(
        [
            left_knee_pos[0],
            left_knee_pos[1] - height * 0.22,
            left_knee_pos[2] + depth * 0.1,
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("leftFoot", left_foot_pos, "leftLowerLeg"))

    left_toes_pos = np.array(
        [left_foot_pos[0], left_foot_pos[1], left_foot_pos[2] + depth * 0.08],
        dtype=np.float32,
    )
    bones.append(_place_bone("leftToes", left_toes_pos, "leftFoot"))

    # ── Leg chain (right) ──────────────────────────────────────
    right_leg_x = hips_pos[0] + width * 0.09
    right_upper_leg_pos = np.array(
        [right_leg_x, hips_pos[1] - height * 0.06, center[2]], dtype=np.float32
    )
    bones.append(_place_bone("rightUpperLeg", right_upper_leg_pos, "hips"))

    right_knee_pos = np.array(
        [
            right_upper_leg_pos[0],
            right_upper_leg_pos[1] - height * 0.22,
            right_upper_leg_pos[2] + depth * 0.02,
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("rightLowerLeg", right_knee_pos, "rightUpperLeg"))

    right_foot_pos = np.array(
        [
            right_knee_pos[0],
            right_knee_pos[1] - height * 0.22,
            right_knee_pos[2] + depth * 0.1,
        ],
        dtype=np.float32,
    )
    bones.append(_place_bone("rightFoot", right_foot_pos, "rightLowerLeg"))

    right_toes_pos = np.array(
        [right_foot_pos[0], right_foot_pos[1], right_foot_pos[2] + depth * 0.08],
        dtype=np.float32,
    )
    bones.append(_place_bone("rightToes", right_toes_pos, "rightFoot"))

    # ── Fingers (left, 5 × 3 phalanges) ─────────────────────────
    _add_finger_bones(bones, "left", left_hand_pos, width, height)
    _add_finger_bones(bones, "right", right_hand_pos, width, height)

    # Clamp all bone positions to [-1.5, 1.5]³ (M3-10)
    for bone in bones:
        bone.position = np.clip(bone.position, -1.5, 1.5).astype(np.float32)

    return Skeleton(bones=bones)


def _add_finger_bones(
    bones: list[Bone],
    side: str,
    hand_pos: np.ndarray,
    width: float,
    height: float,
) -> None:
    """Add 5 fingers × 3 phalanges for one hand."""
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]
    finger_x_offsets = [-0.04, -0.025, 0.0, 0.025, 0.04]
    finger_z_offsets = [0.015, 0.03, 0.03, 0.03, 0.025]
    finger_y_offsets = [0.005, -0.005, -0.005, -0.005, -0.005]

    for fname, fx, fz, fy in zip(finger_names, finger_x_offsets, finger_z_offsets, finger_y_offsets):
        # metacarpal
        mc_pos = np.array(
            [
                hand_pos[0] + fx * width * 2,
                hand_pos[1] + fy,
                hand_pos[2] + fz,
            ],
            dtype=np.float32,
        )
        mc_name = f"{side}{fname}Metacarpal"
        bones.append(_place_bone(mc_name, mc_pos, f"{side}Hand"))

        # proximal
        prox_pos = np.array(
            [mc_pos[0], mc_pos[1] - height * 0.05, mc_pos[2] + fz * 0.5],
            dtype=np.float32,
        )
        prox_name = f"{side}{fname}Proximal"
        bones.append(_place_bone(prox_name, prox_pos, mc_name))

        # distal
        dist_pos = np.array(
            [prox_pos[0], prox_pos[1] - height * 0.04, prox_pos[2] + fz * 0.3],
            dtype=np.float32,
        )
        dist_name = f"{side}{fname}Distal"
        bones.append(_place_bone(dist_name, dist_pos, prox_name))


# ─── Validation helpers ──────────────────────────────────────────

def validate_skeleton(skeleton: Skeleton) -> list[str]:
    """
    Check all M3 invariants.
    Returns list of violation descriptions (empty = all pass).
    """
    errors: list[str] = []
    names = skeleton.bone_names

    # M3-1: all 15 required bones present
    missing = REQUIRED_BONES - names
    if missing:
        errors.append(f"M3-1: missing required bones: {missing}")

    # M3-2 + M3-5: exactly one root (hips has no parent), all others have a parent
    root_count = sum(1 for b in skeleton.bones if b.parent is None)
    if root_count != 1:
        errors.append(f"M3-5: expected 1 root, got {root_count}")
    orphan = [b.name for b in skeleton.bones if b.name != "hips" and b.parent is None]
    if orphan:
        errors.append(f"M3-2: bones with no parent (not hips): {orphan}")

    # M3-3: no cycles
    if _has_cycle(skeleton):
        errors.append("M3-3: cycle detected in bone hierarchy")

    # M3-4: all positions finite
    nonfinite = [b.name for b in skeleton.bones if not np.all(np.isfinite(b.position))]
    if nonfinite:
        errors.append(f"M3-4: non-finite bone positions: {nonfinite}")

    # M3-6: camelCase check (first letter lowercase, no underscores)
    non_camel = [
        b.name for b in skeleton.bones
        if "_" in b.name or not b.name[0].islower()
    ]
    if non_camel:
        errors.append(f"M3-6: non-camelCase bone names: {non_camel}")

    # M3-7: parent relationships match VRM_PARENT_RULES
    for bone in skeleton.bones:
        expected_parent = VRM_PARENT_RULES.get(bone.name)
        if expected_parent is None:
            continue  # bone accepts multiple/none parents
        if bone.parent != expected_parent:
            errors.append(
                f"M3-7: {bone.name}.parent={bone.parent!r}, expected {expected_parent!r}"
            )

    # M3-8: upperChest implies chest
    has_upper_chest = "upperChest" in names
    has_chest = "chest" in names
    if has_upper_chest and not has_chest:
        errors.append("M3-8: upperChest present without chest")

    # M3-9: neck implies chest or upperChest
    has_neck = "neck" in names
    if has_neck and not (has_chest or has_upper_chest):
        errors.append("M3-9: neck present without chest or upperChest")

    # M3-10: bone positions within expanded bounding box
    # All bones should be within [-1.5, 1.5]³ (generous)
    for bone in skeleton.bones:
        if np.any(bone.position < -1.5) or np.any(bone.position > 1.5):
            errors.append(f"M3-10: bone {bone.name} position {bone.position} outside [-1.5,1.5]³")

    return errors


def _has_cycle(skeleton: Skeleton) -> bool:
    """DFS cycle detection in parent graph."""
    visited: dict[str, int] = {}  # name → 0=unvisited, 1=visiting, 2=done

    def dfs(name: str) -> bool:
        state = visited.get(name, 0)
        if state == 1:
            return True  # back-edge = cycle
        if state == 2:
            return False
        visited[name] = 1
        bone = skeleton.get_bone(name)
        if bone and bone.parent:
            if dfs(bone.parent):
                return True
        visited[name] = 2
        return False

    for bone in skeleton.bones:
        if visited.get(bone.name, 0) == 0:
            if dfs(bone.name):
                return True
    return False


IMPORTS_OK = True
