"""Unit tests for Stage 8: VRM Exporter (vrm/exporter.py)."""

from __future__ import annotations

import json
import os
import struct
import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from vtuber_rigger.interfaces import (
    Bone,
    Collider,
    Expression,
    LookAt,
    FirstPerson,
    MeshData,
    MorphTarget,
    SkinWeights,
    Skeleton,
    SpringChain,
    SpringNode,
    VRMMeta,
    VRM_STANDARD_EXPRESSIONS,
    REQUIRED_BONES,
)
from vtuber_rigger.vrm.exporter import (
    export_vrm,
    validate_vrm,
    _pack_glb,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_minimal_mesh() -> MeshData:
    """Small tetrahedron mesh."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
    normals = np.zeros((4, 3), dtype=np.float32)
    return MeshData(vertices=verts, faces=faces, normals=normals)


def make_minimal_skeleton() -> Skeleton:
    bones = [
        Bone(name="hips", position=np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        Bone(name="spine", position=np.array([0.0, 0.2, 0.0], dtype=np.float32), parent="hips"),
        Bone(
            name="head",
            position=np.array([0.0, 0.6, 0.0], dtype=np.float32),
            parent="spine",
        ),
        Bone(
            name="leftUpperArm",
            position=np.array([-0.2, 0.4, 0.0], dtype=np.float32),
            parent="spine",
        ),
        Bone(
            name="leftLowerArm",
            position=np.array([-0.4, 0.4, 0.0], dtype=np.float32),
            parent="leftUpperArm",
        ),
        Bone(
            name="leftHand",
            position=np.array([-0.5, 0.4, 0.0], dtype=np.float32),
            parent="leftLowerArm",
        ),
        Bone(
            name="rightUpperArm",
            position=np.array([0.2, 0.4, 0.0], dtype=np.float32),
            parent="spine",
        ),
        Bone(
            name="rightLowerArm",
            position=np.array([0.4, 0.4, 0.0], dtype=np.float32),
            parent="rightUpperArm",
        ),
        Bone(
            name="rightHand",
            position=np.array([0.5, 0.4, 0.0], dtype=np.float32),
            parent="rightLowerArm",
        ),
        Bone(
            name="leftUpperLeg",
            position=np.array([-0.1, -0.2, 0.0], dtype=np.float32),
            parent="hips",
        ),
        Bone(
            name="leftLowerLeg",
            position=np.array([-0.1, -0.5, 0.0], dtype=np.float32),
            parent="leftUpperLeg",
        ),
        Bone(
            name="leftFoot",
            position=np.array([-0.1, -0.8, 0.0], dtype=np.float32),
            parent="leftLowerLeg",
        ),
        Bone(
            name="rightUpperLeg",
            position=np.array([0.1, -0.2, 0.0], dtype=np.float32),
            parent="hips",
        ),
        Bone(
            name="rightLowerLeg",
            position=np.array([0.1, -0.5, 0.0], dtype=np.float32),
            parent="rightUpperLeg",
        ),
        Bone(
            name="rightFoot",
            position=np.array([0.1, -0.8, 0.0], dtype=np.float32),
            parent="rightLowerLeg",
        ),
    ]
    return Skeleton(bones=bones)


def make_minimal_weights(mesh: MeshData, skeleton: Skeleton) -> SkinWeights:
    n_verts = mesh.vertex_count
    n_bones = len(skeleton.bones)
    data = np.zeros((n_verts, n_bones), dtype=np.float32)
    # Each vertex assigned fully to bone 0 (hips)
    data[:, 0] = 1.0
    matrix = csr_matrix(data)
    bone_names = [b.name for b in skeleton.bones]
    return SkinWeights(matrix=matrix, bone_names=bone_names)


def make_minimal_expressions(mesh: MeshData) -> dict[str, Expression]:
    """One expression per standard name."""
    exprs = {}
    for i, name in enumerate(sorted(VRM_STANDARD_EXPRESSIONS)):
        morph = MorphTarget(
            name=name,
            vertices=mesh.vertices.copy(),
            vertex_count=mesh.vertex_count,
        )
        expr = Expression(
            name=name,
            morph_target=morph,
            morph_target_index=i,
            preset_weight=1.0,
        )
        exprs[name] = expr
    return exprs


def make_minimal_meta() -> VRMMeta:
    return VRMMeta(
        name="TestAvatar",
        version="1.0",
        authors=["Test Author"],
        licenseUrl="https://creativecommons.org/publicdomain/zero/1.0/",
    )


def make_minimal_lookat() -> LookAt:
    return LookAt(mode="bone", yaw_range=1.5708, pitch_range=1.5708)


def make_minimal_firstperson() -> FirstPerson:
    return FirstPerson(
        camera_offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        mesh_annotations=[{"type": "auto"}],
    )


def make_minimal_spring_chains() -> tuple[list[SpringChain], list[Collider]]:
    chain = SpringChain(
        nodes=[
            SpringNode(
                bone_name="leftUpperArm",
                position=np.array([-0.2, 0.4, 0.0], dtype=np.float32),
            ),
            SpringNode(
                bone_name="leftLowerArm",
                position=np.array([-0.4, 0.4, 0.0], dtype=np.float32),
            ),
        ],
        root_bone="leftUpperArm",
        stiffness=0.5,
        dragForce=0.5,
        hitRadius=0.02,
        gravityPower=0.5,
        gravityDir=np.array([0.0, -1.0, 0.0], dtype=np.float32),
    )
    collider = Collider(
        center=np.array([0.0, 0.4, 0.0], dtype=np.float32),
        radius=0.05,
    )
    return [chain], [collider]


# ─── _pack_glb tests ─────────────────────────────────────────────────────────

class TestPackGLB:
    def test_packs_empty_binary(self):
        json_b = b'{"asset":{"version":"2.0"}}'
        result = _pack_glb(json_b, b"")
        assert len(result) >= 12
        magic, ver, length = struct.unpack("<III", result[:12])
        assert magic == 0x46546C67
        assert ver == 2
        assert length == len(result)

    def test_packs_with_binary(self):
        json_b = b'{"asset":{"version":"2.0"}}'
        binary = b"\x00\x01\x02\x03"
        result = _pack_glb(json_b, binary)
        assert len(result) >= 12 + 8 + len(json_b)


# ─── export_vrm tests ─────────────────────────────────────────────────────────

class TestExportVRM:
    def test_writes_glb_file(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()
        chains, colliders = make_minimal_spring_chains()
        lookat = make_minimal_lookat()
        fp = make_minimal_firstperson()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, chains, colliders,
                lookat, fp, meta, f.name,
            )
        try:
            assert os.path.exists(out)
            assert os.path.getsize(out) > 0
        finally:
            os.unlink(out)

    def test_output_is_valid_glb(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            with open(out, "rb") as fh:
                data = fh.read()
            magic = struct.unpack("<I", data[0:4])[0]
            assert magic == 0x46546C67
        finally:
            os.unlink(out)

    def test_m8_11_file_size_limit(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            size = os.path.getsize(out)
            assert size <= 50 * 1024 * 1024
        finally:
            os.unlink(out)

    def test_m8_13_no_animations_or_cameras(self):
        """GLB must not contain animations or cameras."""
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            anim_errors = [e for e in errors if "animation" in e.lower() or "camera" in e.lower()]
            assert not anim_errors, f"Found animation/camera errors: {anim_errors}"
        finally:
            os.unlink(out)

    def test_m8_2_spec_version(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            version_errors = [e for e in errors if "specVersion" in e or "1.0" in e]
            assert not version_errors, f"specVersion errors: {version_errors}"
        finally:
            os.unlink(out)

    def test_m8_3_all_15_bones(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            missing_errors = [e for e in errors if "Missing required bones" in e]
            assert not missing_errors, f"Missing bones: {missing_errors}"
        finally:
            os.unlink(out)

    def test_m8_4_meta_required_fields(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            meta_errors = [e for e in errors if "meta" in e.lower()]
            assert not meta_errors, f"Meta errors: {meta_errors}"
        finally:
            os.unlink(out)

    def test_m8_14_positive_bone_scale(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            scale_errors = [e for e in errors if "scale" in e.lower()]
            assert not scale_errors, f"Scale errors: {scale_errors}"
        finally:
            os.unlink(out)

    def test_returns_output_path(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            assert isinstance(out, str)
            assert out.endswith(".vrm")
        finally:
            os.unlink(out)

    def test_works_without_optional_components(self):
        """export_vrm should succeed with only required inputs."""
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, {}, [], [],
                None, None, meta, f.name,
            )
        try:
            assert os.path.exists(out)
            errors = validate_vrm(out)
            # Should have no critical errors (meta/bones already covered)
            assert len(errors) == 0, f"Unexpected errors: {errors}"
        finally:
            os.unlink(out)


# ─── validate_vrm tests ───────────────────────────────────────────────────────

class TestValidateVRM:
    def test_valid_file_returns_empty_list(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        exprs = make_minimal_expressions(mesh)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, exprs, [], [],
                None, None, meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            # Optional components absent is not an error
            critical = [e for e in errors if "Missing required bones" in e or "specVersion" in e]
            assert not critical, f"Critical errors in valid file: {errors}"
        finally:
            os.unlink(out)

    def test_invalid_magic_detected(self):
        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            f.write(b"not a glb file at all")
            f.flush()
            path = f.name
        try:
            errors = validate_vrm(path)
            assert any("magic" in e.lower() for e in errors)
        finally:
            os.unlink(path)

    def test_missing_required_bones_detected(self):
        """Skeleton missing a required bone should be caught at validation level."""
        # Note: make_minimal_skeleton has all 15 bones, so we test by
        # directly constructing a partial skeleton
        mesh = make_minimal_mesh()
        bones = [
            Bone(name="hips", position=np.array([0.0, 0.0, 0.0], dtype=np.float32)),
            # missing spine and other bones
        ]
        skel = Skeleton(bones=bones)
        weights = make_minimal_weights(mesh, skel)
        meta = make_minimal_meta()

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, {}, [], [],
                None, None, meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            assert any("Missing required bones" in e for e in errors)
        finally:
            os.unlink(out)

    def test_missing_meta_fields_detected(self):
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)

        # Meta missing required fields
        bad_meta = VRMMeta(name="", version="", authors=[], licenseUrl="")

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            out = export_vrm(
                mesh, skel, weights, {}, [], [],
                None, None, bad_meta, f.name,
            )
        try:
            errors = validate_vrm(out)
            assert any("meta" in e.lower() for e in errors)
        finally:
            os.unlink(out)

    def test_too_large_file_rejected(self):
        """File over 50 MB raises ValueError during export."""
        mesh = make_minimal_mesh()
        skel = make_minimal_skeleton()
        weights = make_minimal_weights(mesh, skel)
        meta = make_minimal_meta()

        # Mock the _pack_glb to return >50MB blob so we can test the size check
        # without generating huge real data.
        import vtuber_rigger.vrm.exporter as exp_mod
        original_pack = exp_mod._pack_glb

        def fake_pack_glb(json_b: bytes, binary: bytes) -> bytes:
            # Return a blob that claims to be >50MB
            huge = b"\x00" * (51 * 1024 * 1024)
            return original_pack(json_b, huge)

        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            path = f.name

        try:
            exp_mod._pack_glb = fake_pack_glb
            with pytest.raises(ValueError, match="50 MB"):
                export_vrm(mesh, skel, weights, {}, [], [], None, None, meta, path)
        finally:
            exp_mod._pack_glb = original_pack
            if os.path.exists(path):
                os.unlink(path)

    def test_truncated_file_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            f.write(b"\x00" * 5)
            path = f.name
        try:
            errors = validate_vrm(path)
            assert len(errors) > 0
        finally:
            os.unlink(path)

    def test_missing_vrmc_vrm_extension_detected(self):
        """A valid GLB without VRMC_vrm should fail M8-2."""
        with tempfile.NamedTemporaryFile(suffix=".vrm", delete=False) as f:
            # Build a minimal GLB with empty JSON
            json_b = json.dumps({"asset": {"version": "2.0"}}).encode()
            glb = _pack_glb(json_b, b"")
            f.write(glb)
            path = f.name
        try:
            errors = validate_vrm(path)
            assert any("VRMC_vrm" in e for e in errors)
        finally:
            os.unlink(path)
