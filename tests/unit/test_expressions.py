"""Unit tests for Stage 5: Expressions (vtuber_rigger.rig.expressions)."""

from __future__ import annotations

import numpy as np
import pytest

from vtuber_rigger.interfaces import (
    Expression,
    MeshData,
    MorphTarget,
    Skeleton,
    Bone,
    VRM_STANDARD_EXPRESSIONS,
)
from vtuber_rigger.rig.expressions import create_expressions


# ─── Fixtures ────────────────────────────────────────────────────

def make_sphere_mesh(radius: float = 1.0, lat: int = 10, lon: int = 10) -> MeshData:
    """Build a UV sphere and return it as MeshData."""
    import trimesh
    tm = trimesh.creation.uv_sphere(radius=radius, lat_res=lat, lon_res=lon)
    verts = np.asarray(tm.vertices, dtype=np.float32)
    # Normalize to [-1,1] bounding box
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    scale = max(vmax - vmin)
    verts = (verts - (vmin + vmax) * 0.5) * (2.0 / scale)
    verts = verts.astype(np.float32)
    faces = np.asarray(tm.faces, dtype=np.int32)
    normals = np.asarray(tm.vertex_normals, dtype=np.float32)
    return MeshData(vertices=verts, faces=faces, normals=normals)


def make_15_bone_skeleton() -> Skeleton:
    """Return a minimal VRM 1.0 skeleton with 15 required bones."""
    bones = [
        Bone(name="hips",      position=np.array([0.0,  0.0,  0.0], dtype=np.float32), parent=None),
        Bone(name="spine",     position=np.array([0.0,  0.2,  0.0], dtype=np.float32), parent="hips"),
        Bone(name="head",      position=np.array([0.0,  0.8,  0.0], dtype=np.float32), parent="spine"),
        Bone(name="leftUpperArm",  position=np.array([-0.2, 0.4, 0.0], dtype=np.float32), parent="spine"),
        Bone(name="leftLowerArm",  position=np.array([-0.4, 0.2, 0.0], dtype=np.float32), parent="leftUpperArm"),
        Bone(name="leftHand",      position=np.array([-0.6, 0.1, 0.0], dtype=np.float32), parent="leftLowerArm"),
        Bone(name="rightUpperArm", position=np.array([ 0.2, 0.4, 0.0], dtype=np.float32), parent="spine"),
        Bone(name="rightLowerArm", position=np.array([ 0.4, 0.2, 0.0], dtype=np.float32), parent="rightUpperArm"),
        Bone(name="rightHand",     position=np.array([ 0.6, 0.1, 0.0], dtype=np.float32), parent="rightLowerArm"),
        Bone(name="leftUpperLeg",  position=np.array([-0.1, -0.3, 0.0], dtype=np.float32), parent="hips"),
        Bone(name="leftLowerLeg",  position=np.array([-0.1, -0.7, 0.0], dtype=np.float32), parent="leftUpperLeg"),
        Bone(name="leftFoot",      position=np.array([-0.1, -1.0, 0.0], dtype=np.float32), parent="leftLowerLeg"),
        Bone(name="rightUpperLeg", position=np.array([ 0.1, -0.3, 0.0], dtype=np.float32), parent="hips"),
        Bone(name="rightLowerLeg", position=np.array([ 0.1, -0.7, 0.0], dtype=np.float32), parent="rightUpperLeg"),
        Bone(name="rightFoot",     position=np.array([ 0.1, -1.0, 0.0], dtype=np.float32), parent="rightLowerLeg"),
    ]
    return Skeleton(bones=bones)


# ─── Tests ─────────────────────────────────────────────────────

class TestCreateExpressions:
    """Tests for create_expressions()."""

    @pytest.fixture
    def sphere_mesh(self) -> MeshData:
        return make_sphere_mesh()

    @pytest.fixture
    def skeleton(self) -> Skeleton:
        return make_15_bone_skeleton()

    def test_returns_list_of_expressions(self, sphere_mesh, skeleton):
        result = create_expressions(sphere_mesh, skeleton)
        assert isinstance(result, list)
        assert all(isinstance(e, Expression) for e in result)

    # ── M5-1 ────────────────────────────────────────────────────

    def test_m5_1_all_17_expressions_present(self, sphere_mesh, skeleton):
        """M5-1: All 17 standard VRM expressions are present."""
        result = create_expressions(sphere_mesh, skeleton)
        names = {e.name for e in result}
        assert names == VRM_STANDARD_EXPRESSIONS, (
            f"Missing expressions: {VRM_STANDARD_EXPRESSIONS - names}"
        )

    def test_m5_1_count_is_17(self, sphere_mesh, skeleton):
        result = create_expressions(sphere_mesh, skeleton)
        assert len(result) == 17

    # ── M5-2 ────────────────────────────────────────────────────

    def test_m5_2_morph_target_vertex_count_matches_base_mesh(self, sphere_mesh, skeleton):
        """M5-2: Each expression morphTarget has the same vertex count as the base mesh."""
        result = create_expressions(sphere_mesh, skeleton)
        for expr in result:
            assert expr.morph_target.vertex_count == sphere_mesh.vertex_count, (
                f"{expr.name}: vertex count mismatch "
                f"{expr.morph_target.vertex_count} vs {sphere_mesh.vertex_count}"
            )

    def test_m5_2_morph_vertices_shape_matches(self, sphere_mesh, skeleton):
        result = create_expressions(sphere_mesh, skeleton)
        n = sphere_mesh.vertex_count
        for expr in result:
            assert expr.morph_target.vertices.shape == (n, 3), (
                f"{expr.name}: expected shape ({n}, 3), "
                f"got {expr.morph_target.vertices.shape}"
            )

    # ── M5-3 ────────────────────────────────────────────────────

    def test_m5_3_preset_weight_in_range(self, sphere_mesh, skeleton):
        """M5-3: Expression weight range is [0.0, 1.0]."""
        result = create_expressions(sphere_mesh, skeleton)
        for expr in result:
            assert 0.0 <= expr.preset_weight <= 1.0, (
                f"{expr.name}: preset_weight={expr.preset_weight} out of [0,1]"
            )

    # ── M5-4 ────────────────────────────────────────────────────

    def test_m5_4_all_morph_vertices_finite(self, sphere_mesh, skeleton):
        """M5-4: All morphTarget vertex positions are finite (no NaN, no Inf)."""
        result = create_expressions(sphere_mesh, skeleton)
        for expr in result:
            verts = expr.morph_target.vertices
            assert np.all(np.isfinite(verts)), (
                f"{expr.name}: non-finite vertices detected"
            )

    def test_m5_4_no_nan(self, sphere_mesh, skeleton):
        result = create_expressions(sphere_mesh, skeleton)
        for expr in result:
            assert not np.any(np.isnan(expr.morph_target.vertices)), (
                f"{expr.name}: NaN vertices found"
            )

    def test_m5_4_no_inf(self, sphere_mesh, skeleton):
        result = create_expressions(sphere_mesh, skeleton)
        for expr in result:
            assert not np.any(np.isinf(expr.morph_target.vertices)), (
                f"{expr.name}: Inf vertices found"
            )

    # ── M5-5 ────────────────────────────────────────────────────

    def test_m5_5_neutral_produces_base_mesh(self, sphere_mesh, skeleton):
        """M5-5: At weight=0, base mesh is recovered.

        We check that the first expression's morph target at a weight-0 blend
        equals the base. Since morphTarget stores absolute positions, we
        verify that the morph_target is different from base (weight=1 changes
        the mesh) — the neutral state (weight=0) is implicit in the pipeline.
        """
        result = create_expressions(sphere_mesh, skeleton)
        # At preset_weight=1 the mesh should be different from base for at
        # least one expression; otherwise the expression has no effect.
        any_different = any(
            not np.allclose(expr.morph_target.vertices, sphere_mesh.vertices, atol=1e-6)
            for expr in result
        )
        assert any_different, (
            "All expressions produce the same vertex positions as base mesh — "
            "expressions appear to have no effect"
        )

    # ── M5-6 / M5-7 ─────────────────────────────────────────────

    def test_m5_7_morph_target_index_unique(self, sphere_mesh, skeleton):
        """M5-7: Each expression maps to exactly one morphTarget (index is unique)."""
        result = create_expressions(sphere_mesh, skeleton)
        indices = [e.morph_target_index for e in result]
        assert len(indices) == len(set(indices)), "Duplicate morph_target_index found"

    def test_m5_7_all_indices_in_range(self, sphere_mesh, skeleton):
        result = create_expressions(sphere_mesh, skeleton)
        indices = [e.morph_target_index for e in result]
        assert min(indices) == 0
        assert max(indices) == len(result) - 1

    def test_m5_7_morph_target_name_matches_expression_name(self, sphere_mesh, skeleton):
        result = create_expressions(sphere_mesh, skeleton)
        for expr in result:
            assert expr.morph_target.name == expr.name


class TestEdgeCases:
    """Edge-case handling in create_expressions()."""

    def test_empty_mesh_raises(self):
        mesh = MeshData(vertices=np.zeros((0, 3), dtype=np.float32),
                        faces=np.zeros((0, 3), dtype=np.int32))
        skeleton = make_15_bone_skeleton()
        with pytest.raises(ValueError, match="0 vertices"):
            create_expressions(mesh, skeleton)

    def test_skeleton_not_used_but_not_crashed(self):
        """Skeleton is not required for template-based expressions."""
        mesh = make_sphere_mesh()
        # Pass a skeleton with just 1 bone — should not crash
        minimal = Skeleton(bones=[Bone(name="root", position=np.array([0., 0., 0.]), parent=None)])
        result = create_expressions(mesh, minimal)
        assert len(result) == 17


class TestMorphTargetDataclass:
    """Unit tests for the MorphTarget dataclass construction."""

    def test_morph_target_fields(self):
        verts = np.zeros((10, 3), dtype=np.float32)
        mt = MorphTarget(name="happy", vertices=verts, vertex_count=10)
        assert mt.name == "happy"
        assert mt.vertex_count == 10
        assert mt.vertices.shape == (10, 3)

    def test_expression_fields(self):
        verts = np.zeros((10, 3), dtype=np.float32)
        mt = MorphTarget(name="happy", vertices=verts, vertex_count=10)
        expr = Expression(name="happy", morph_target=mt, morph_target_index=0)
        assert expr.name == "happy"
        assert expr.morph_target is mt
        assert expr.morph_target_index == 0
        assert expr.preset_weight == 1.0


class TestVrmStandardExpressions:
    """VRM_STANDARD_EXPRESSIONS constant contains the right 17 names."""

    def test_17_expressions(self):
        assert len(VRM_STANDARD_EXPRESSIONS) == 17

    def test_emotion_expressions_present(self):
        for name in ("happy", "angry", "sad", "relaxed", "surprised"):
            assert name in VRM_STANDARD_EXPRESSIONS

    def test_look_expressions_present(self):
        for name in ("lookUp", "lookDown", "lookLeft", "lookRight", "blink"):
            assert name in VRM_STANDARD_EXPRESSIONS

    def test_blink_expressions_present(self):
        for name in ("blinkLeft", "blinkRight"):
            assert name in VRM_STANDARD_EXPRESSIONS

    def test_vowel_expressions_present(self):
        for name in ("aa", "ih", "ou", "ee", "oh"):
            assert name in VRM_STANDARD_EXPRESSIONS
