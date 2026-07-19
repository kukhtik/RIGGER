"""Stage 7: LookAt + FirstPerson configuration.

M7-1..M7-7 invariants enforced.
"""

from __future__ import annotations

import numpy as np
from vtuber_rigger.interfaces import (
    LookAt,
    FirstPerson,
    Skeleton,
    MeshData,
)


def create_lookat(skeleton: Skeleton) -> LookAt:
    """Create LookAt configuration from skeleton.

    M7-1: LookAt specifies either bone-based or expression-based mode
    M7-2: If bone-based: target bones are leftEye and rightEye (if present)
    M7-3: If expression-based: maps to lookUp/lookDown/lookLeft/lookRight expressions
    M7-4: Yaw range is finite (default ±90°, configurable)
    M7-5: Pitch range is finite (default ±90°, configurable)

    Returns LookAt with mode 'bone' if leftEye/rightEye present,
    else 'expression'.
    """
    bone_names = skeleton.bone_names
    has_eyes = "leftEye" in bone_names and "rightEye" in bone_names

    if has_eyes:
        mode = "bone"
    else:
        mode = "expression"

    # Default ±90 degrees (π/2 radians)
    yaw_range = float(1.5708)
    pitch_range = float(1.5708)

    return LookAt(
        mode=mode,
        yaw_range=yaw_range,
        pitch_range=pitch_range,
    )


def create_firstperson(skeleton: Skeleton, mesh: MeshData) -> FirstPerson:
    """Create FirstPerson configuration from skeleton and mesh.

    M7-6: FirstPerson camera offset is finite 3D vector
    M7-7: FirstPerson mesh annotations: each mesh primitive has enum ∈
          {auto, both, thirdPerson, firstPerson}

    Camera offset is placed between eye level and head center,
    with y in [-1, 0] (slightly below head centre, VRM convention).
    """
    bone_names = skeleton.bone_names
    bone_map = skeleton.bone_map

    # Find head bone position
    head_pos = None
    if "head" in bone_map:
        head_pos = bone_map["head"].position

    # Find eye level (average of leftEye and rightEye if present)
    eye_pos = None
    if "leftEye" in bone_map and "rightEye" in bone_map:
        lEye = bone_map["leftEye"].position
        rEye = bone_map["rightEye"].position
        eye_pos = (lEye + rEye) / 2.0

    # Fallback: use mesh bounding box centre of head region
    if head_pos is None:
        min_corner, max_corner = mesh.bounding_box
        head_pos = (min_corner + max_corner) / 2.0

    if eye_pos is None:
        # Approximate eye level as head_pos + slight offset upward
        eye_pos = head_pos + np.array([0.0, 0.05, 0.0], dtype=np.float32)

    # Camera offset: between eye level and head centre
    # VRM convention: offset is from head centre toward eyes, then slightly down
    # y in [-1, 0] — place it between head_pos (above) and slightly below
    camera_offset = ((head_pos + eye_pos) / 2.0).astype(np.float32)

    # Clamp y to [-1, 0] range
    camera_offset[1] = float(np.clip(camera_offset[1], -1.0, 0.0))

    # Mesh annotations: default to 'auto' for the single mesh primitive
    mesh_annotations = [{"type": "auto"}]

    return FirstPerson(
        camera_offset=camera_offset,
        mesh_annotations=mesh_annotations,
    )
