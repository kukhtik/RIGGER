"""Unit tests for Stage 3: Skeleton Prediction (vtuber_rigger.rig.skeleton)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from vtuber_rigger.interfaces import (
    REQUIRED_BONES,
    VRM_PARENT_RULES,
    Bone,
    MeshData,
    Skeleton,
)
from vtuber_rigger.rig.skeleton import (
    IMPORTS_OK,
    _add_finger_bones,
    _detect_joint_by_curvature,
    _has_cycle,
    predict_skeleton,
    validate_skeleton,
)


# ─── Test mesh fixture ─────────────────────────────────────────────

def make_sphere_mesh_data(n_vertices: int = 200) -> MeshData:
    """
    Build a normalized triangle sphere mesh with `n_vertices`.
    Returns a MeshData with normals and faces but no labels.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3)
    verts = np.asarray(sphere.vertices, dtype=np.float32)
    # Normalize to [-1, 1]³
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


# ─── Test predict_skeleton ─────────────────────────────────────────

class TestPredictSkeleton:
    def test_returns_skeleton_instance(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        assert isinstance(skel, Skeleton)

    def test_all_15_required_bones_present(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        names = skel.bone_names
        missing = REQUIRED_BONES - names
        assert not missing, f"Missing required bones: {missing}"

    def test_hips_is_root(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        hips = skel.get_bone("hips")
        assert hips is not None
        assert hips.parent is None

    def test_no_cycles(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        assert not _has_cycle(skel)

    def test_all_bone_positions_finite(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        for bone in skel.bones:
            assert np.all(np.isfinite(bone.position)), f"Non-finite position for {bone.name}"

    def test_bone_names_are_camel_case(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        for bone in skel.bones:
            name = bone.name
            # Must start with lowercase and contain no underscores
            assert name[0].islower(), f"Bone {name} does not start with lowercase"
            assert "_" not in name, f"Bone {name} contains underscore (not camelCase)"

    def test_upper_chest_implies_chest(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        if "upperChest" in skel.bone_names:
            assert "chest" in skel.bone_names, "upperChest present without chest"

    def test_neck_implies_chest_or_upperChest(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        if "neck" in skel.bone_names:
            has_intermediate = "chest" in skel.bone_names or "upperChest" in skel.bone_names
            assert has_intermediate, "neck present without chest or upperChest"

    def test_spine_parent_is_hips(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        spine = skel.get_bone("spine")
        assert spine is not None
        assert spine.parent == "hips"

    def test_head_parent_valid(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        head = skel.get_bone("head")
        assert head is not None
        assert head.parent in {"neck", "upperChest", "chest", "spine", None}

    def test_arm_parent_relationships(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        for side in ("left", "right"):
            upper = skel.get_bone(f"{side}UpperArm")
            lower = skel.get_bone(f"{side}LowerArm")
            hand = skel.get_bone(f"{side}Hand")
            assert upper is not None
            assert lower is not None
            assert hand is not None
            assert lower.parent == f"{side}UpperArm"
            assert hand.parent == f"{side}LowerArm"
            assert upper.parent in {"upperChest", "chest", "spine"}

    def test_leg_parent_relationships(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        for side in ("left", "right"):
            upper = skel.get_bone(f"{side}UpperLeg")
            lower = skel.get_bone(f"{side}LowerLeg")
            foot = skel.get_bone(f"{side}Foot")
            assert upper is not None
            assert lower is not None
            assert foot is not None
            assert lower.parent == f"{side}UpperLeg"
            assert foot.parent == f"{side}LowerLeg"
            assert upper.parent == "hips"

    def test_bone_count_at_least_15(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        assert len(skel.bones) >= 15

    def test_validate_skeleton_returns_empty_for_valid_skeleton(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        errors = validate_skeleton(skel)
        assert errors == [], f"M3 validation errors: {errors}"

    def test_bones_within_expanded_bounds(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        for bone in skel.bones:
            assert np.all(bone.position >= -1.5), f"{bone.name} position {bone.position} below -1.5"
            assert np.all(bone.position <= 1.5), f"{bone.name} position {bone.position} above 1.5"


class TestDetectJointByCurvature:
    def test_returns_finite_point(self):
        mesh = make_sphere_mesh_data()
        start = mesh.vertices[0].copy()
        direction = np.array([0.1, 1.0, 0.0], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        joint = _detect_joint_by_curvature(mesh, start, direction, max_t=0.5)
        assert np.all(np.isfinite(joint))
        assert joint.shape == (3,)


class TestHasCycle:
    def test_no_cycle_for_valid_skeleton(self):
        mesh = make_sphere_mesh_data()
        skel = predict_skeleton(mesh)
        assert not _has_cycle(skel)

    def test_cycle_detected_in_broken_skeleton(self):
        # Build a skeleton with a deliberate cycle
        bones = [
            Bone(name="hips", position=np.array([0.0, 0.0, 0.0], dtype=np.float32), parent=None),
            Bone(name="spine", position=np.array([0.0, 0.3, 0.0], dtype=np.float32), parent="hips"),
            Bone(name="chest", position=np.array([0.0, 0.6, 0.0], dtype=np.float32), parent="spine"),
            # Deliberate cycle: spine's parent points back
            Bone(name="hips2", position=np.array([0.0, 0.0, 0.0], dtype=np.float32), parent="chest"),
            Bone(name="spine2", position=np.array([0.0, 0.3, 0.0], dtype=np.float32), parent="hips2"),
        ]
        # We don't use this directly — just test the has_cycle helper on a cycle
        # Build a minimal skeleton with cycle: a -> b -> c -> a
        cycle_bones = [
            Bone(name="a", position=np.array([0.0, 0.0, 0.0], dtype=np.float32), parent="c"),
            Bone(name="b", position=np.array([0.0, 0.3, 0.0], dtype=np.float32), parent="a"),
            Bone(name="c", position=np.array([0.0, 0.6, 0.0], dtype=np.float32), parent="b"),
        ]
        skel_cycle = Skeleton(bones=cycle_bones)
        assert _has_cycle(skel_cycle), "Cycle not detected in a->b->c->a cycle"


class TestAddFingerBones:
    def test_adds_5_fingers_3_phalanges_each(self):
        bones: list[Bone] = []
        hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _add_finger_bones(bones, "left", hand_pos, width=1.0, height=1.0)
        assert len(bones) == 15  # 5 fingers × 3 phalanges
        names = {b.name for b in bones}
        for fname in ["Thumb", "Index", "Middle", "Ring", "Little"]:
            assert f"left{fname}Metacarpal" in names
            assert f"left{fname}Proximal" in names
            assert f"left{fname}Distal" in names

    def test_finger_parent_chain_correct(self):
        bones: list[Bone] = []
        hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        _add_finger_bones(bones, "right", hand_pos, width=1.0, height=1.0)
        skel = Skeleton(bones=bones)
        for fname in ["Thumb", "Index", "Middle", "Ring", "Little"]:
            mc = skel.get_bone(f"right{fname}Metacarpal")
            prox = skel.get_bone(f"right{fname}Proximal")
            dist = skel.get_bone(f"right{fname}Distal")
            assert mc is not None
            assert prox is not None
            assert dist is not None
            assert prox.parent == f"right{fname}Metacarpal"
            assert dist.parent == f"right{fname}Proximal"


class TestSentinel:
    def test_imports_ok_true(self):
        assert IMPORTS_OK is True
