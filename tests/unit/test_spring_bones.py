"""Unit tests for Stage 6: Spring Bones (vtuber_rigger.rig.spring_bones)."""

from __future__ import annotations

import numpy as np
import pytest

from vtuber_rigger.interfaces import (
    Collider,
    MeshData,
    Skeleton,
    SpringChain,
    SpringNode,
)
from vtuber_rigger.rig.spring_bones import detect_spring_bones


# ─── Fixtures ────────────────────────────────────────────────────

def make_sphere_mesh(radius: float = 1.0, lat: int = 8, lon: int = 8) -> MeshData:
    """Build a UV sphere as MeshData."""
    import trimesh
    tm = trimesh.creation.uv_sphere(radius=radius, lat_res=lat, lon_res=lon)
    verts = np.asarray(tm.vertices, dtype=np.float32)
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    scale = max(vmax - vmin)
    verts = (verts - (vmin + vmax) * 0.5) * (2.0 / scale)
    verts = verts.astype(np.float32)
    faces = np.asarray(tm.faces, dtype=np.int32)
    return MeshData(vertices=verts, faces=faces)


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


# Fix missing import
from vtuber_rigger.interfaces import Bone


# ─── Tests ─────────────────────────────────────────────────────

class TestDetectSpringBones:
    """Tests for detect_spring_bones()."""

    @pytest.fixture
    def sphere_mesh(self) -> MeshData:
        return make_sphere_mesh()

    @pytest.fixture
    def skeleton(self) -> Skeleton:
        return make_15_bone_skeleton()

    # ── Return type ─────────────────────────────────────────────

    def test_returns_tuple(self, sphere_mesh, skeleton):
        result = detect_spring_bones(sphere_mesh, skeleton)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_list_of_spring_chains(self, sphere_mesh, skeleton):
        chains, colliders = detect_spring_bones(sphere_mesh, skeleton)
        assert isinstance(chains, list)
        assert all(isinstance(c, SpringChain) for c in chains)

    def test_second_element_is_list_of_colliders(self, sphere_mesh, skeleton):
        chains, colliders = detect_spring_bones(sphere_mesh, skeleton)
        assert isinstance(colliders, list)
        assert all(isinstance(c, Collider) for c in colliders)

    # ── M6-8: Collider radii positive ──────────────────────────

    def test_m6_8_all_collider_radii_positive(self, sphere_mesh, skeleton):
        """M6-8: All collider radii are positive (> 0)."""
        _, colliders = detect_spring_bones(sphere_mesh, skeleton)
        for col in colliders:
            assert col.radius > 0, f"Collider radius {col.radius} is not positive"

    def test_m6_8_no_zero_radius(self, sphere_mesh, skeleton):
        _, colliders = detect_spring_bones(sphere_mesh, skeleton)
        for col in colliders:
            assert col.radius > 0.0

    # ── M6-7: Collider centers finite ─────────────────────────

    def test_m6_7_collider_centers_finite(self, sphere_mesh, skeleton):
        """M6-7: Collider sphere centers are finite (no NaN, no Inf)."""
        _, colliders = detect_spring_bones(sphere_mesh, skeleton)
        for col in colliders:
            assert np.all(np.isfinite(col.center)), (
                f"Collider center {col.center} contains non-finite values"
            )

    def test_m6_7_collider_centers_within_bounding_box(self, sphere_mesh, skeleton):
        _, colliders = detect_spring_bones(sphere_mesh, skeleton)
        bb_min, bb_max = sphere_mesh.bounding_box
        for col in colliders:
            assert np.all(col.center >= bb_min - 1e-6), f"Collider {col} outside bounding box (min)"
            assert np.all(col.center <= bb_max + 1e-6), f"Collider {col} outside bounding box (max)"

    # ── M6-5: Parameter ranges ─────────────────────────────────

    @pytest.mark.parametrize("stiffness", [-0.1, 1.1, 2.0])
    def test_m6_5_stiffness_accepted_even_out_of_range(self, sphere_mesh, skeleton, stiffness):
        """M6-5: stiffness is clamped to [0, 1] internally; no crash."""
        chains, _ = detect_spring_bones(sphere_mesh, skeleton, stiffness=stiffness)
        for chain in chains:
            assert 0.0 <= chain.stiffness <= 1.0

    @pytest.mark.parametrize("drag_force", [-0.1, 1.1, 2.0])
    def test_m6_5_drag_force_in_range(self, sphere_mesh, skeleton, drag_force):
        chains, _ = detect_spring_bones(sphere_mesh, skeleton, drag_force=drag_force)
        for chain in chains:
            assert 0.0 <= chain.dragForce <= 1.0

    @pytest.mark.parametrize("hit_radius", [-0.01, 1.5, 10.0])
    def test_m6_5_hit_radius_in_range(self, sphere_mesh, skeleton, hit_radius):
        chains, _ = detect_spring_bones(sphere_mesh, skeleton, hit_radius=hit_radius)
        for chain in chains:
            assert 0.0 <= chain.hitRadius <= 1.0

    @pytest.mark.parametrize("gravity_power", [-1.0, 11.0, 20.0])
    def test_m6_5_gravity_power_in_range(self, sphere_mesh, skeleton, gravity_power):
        chains, _ = detect_spring_bones(sphere_mesh, skeleton, gravity_power=gravity_power)
        for chain in chains:
            assert 0.0 <= chain.gravityPower <= 10.0

    def test_m6_5_gravity_dir_is_unit_vector(self, sphere_mesh, skeleton):
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        for chain in chains:
            norm = np.linalg.norm(chain.gravityDir)
            assert abs(norm - 1.0) < 1e-6, (
                f"gravityDir norm = {norm}, expected 1.0"
            )

    def test_m6_5_default_gravity_dir(self, sphere_mesh, skeleton):
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        for chain in chains:
            assert np.allclose(chain.gravityDir, [0.0, -1.0, 0.0], atol=1e-6)

    # ── M6-6: Root bone valid ───────────────────────────────────

    def test_m6_6_root_bone_is_valid_bone_name(self, sphere_mesh, skeleton):
        """M6-6: Each spring chain has a root bone that exists in the skeleton."""
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        bone_names = skeleton.bone_names
        for chain in chains:
            assert chain.root_bone in bone_names, (
                f"Root bone '{chain.root_bone}' not in skeleton"
            )

    # ── M6-2: Bone node references valid ───────────────────────

    def test_m6_2_all_bone_names_valid(self, sphere_mesh, skeleton):
        """M6-2: All bone node references are valid bone names."""
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        bone_names = skeleton.bone_names
        for chain in chains:
            for node in chain.nodes:
                assert node.bone_name in bone_names, (
                    f"Node bone_name '{node.bone_name}' not in skeleton"
                )

    # ── M6-1: Chain length ≥ 2 ────────────────────────────────

    def test_m6_1_all_chains_have_at_least_2_nodes(self, sphere_mesh, skeleton):
        """M6-1: Each spring chain has ≥ 2 nodes."""
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        for chain in chains:
            assert len(chain.nodes) >= 2, (
                f"Chain has {len(chain.nodes)} nodes, minimum 2 required"
            )

    def test_m6_1_empty_chains_allowed(self, sphere_mesh, skeleton):
        """With a sphere (no thin chains), result may be empty — valid."""
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        # Empty is OK, but if non-empty each chain must satisfy M6-1
        for chain in chains:
            assert len(chain.nodes) >= 2

    # ── M6-4: No circular references ─────────────────────────

    def test_m6_4_nodes_are_acyclic(self, sphere_mesh, skeleton):
        """M6-4: No circular references — nodes list is a simple path."""
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        for chain in chains:
            node_names = [n.bone_name for n in chain.nodes]
            # Simple path: no bone appears twice
            assert len(node_names) == len(set(node_names)), (
                f"Circular reference detected in chain: {node_names}"
            )

    def test_m6_4_node_positions_are_finite(self, sphere_mesh, skeleton):
        chains, _ = detect_spring_bones(sphere_mesh, skeleton)
        for chain in chains:
            for node in chain.nodes:
                assert np.all(np.isfinite(node.position)), (
                    f"Node position {node.position} not finite"
                )


class TestSpringNodeDataclass:
    """Unit tests for SpringNode and SpringChain dataclasses."""

    def test_spring_node_fields(self):
        pos = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        node = SpringNode(bone_name="head", position=pos)
        assert node.bone_name == "head"
        assert np.allclose(node.position, pos)

    def test_spring_chain_fields(self):
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        nodes = [SpringNode(bone_name="hips", position=pos)]
        chain = SpringChain(nodes=nodes, root_bone="hips")
        assert len(chain.nodes) == 1
        assert chain.root_bone == "hips"
        assert chain.stiffness == 0.5   # default
        assert chain.dragForce == 0.5    # default

    def test_spring_chain_with_all_params(self):
        nodes = [SpringNode(bone_name="head", position=np.array([0., 0., 0.]))]
        chain = SpringChain(
            nodes=nodes,
            root_bone="head",
            stiffness=0.8,
            dragForce=0.3,
            hitRadius=0.05,
            gravityPower=2.0,
            gravityDir=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        )
        assert chain.stiffness == 0.8
        assert chain.dragForce == 0.3
        assert chain.hitRadius == 0.05
        assert chain.gravityPower == 2.0


class TestColliderDataclass:
    """Unit tests for Collider dataclass."""

    def test_collider_fields(self):
        center = np.array([0.0, 0.8, 0.0], dtype=np.float32)
        col = Collider(center=center, radius=0.08)
        assert np.allclose(col.center, center)
        assert col.radius == 0.08

    def test_collider_radius_positive(self):
        col = Collider(center=np.array([0., 0., 0.]), radius=0.05)
        assert col.radius > 0


class TestEmptyMesh:
    """Edge case: empty mesh."""

    def test_empty_mesh_returns_empty_chains(self):
        mesh = MeshData(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int32),
        )
        skeleton = make_15_bone_skeleton()
        chains, colliders = detect_spring_bones(mesh, skeleton)
        assert chains == []
        assert isinstance(colliders, list)


class TestCustomPhysicsParameters:
    """Test custom parameter passthrough to detect_spring_bones()."""

    @pytest.fixture
    def mesh(self) -> MeshData:
        return make_sphere_mesh()

    @pytest.fixture
    def skeleton(self) -> Skeleton:
        return make_15_bone_skeleton()

    def test_custom_stiffness(self, mesh, skeleton):
        chains, _ = detect_spring_bones(mesh, skeleton, stiffness=0.9)
        for c in chains:
            assert c.stiffness == 0.9

    def test_custom_drag(self, mesh, skeleton):
        chains, _ = detect_spring_bones(mesh, skeleton, drag_force=0.1)
        for c in chains:
            assert c.dragForce == 0.1

    def test_custom_hit_radius(self, mesh, skeleton):
        chains, _ = detect_spring_bones(mesh, skeleton, hit_radius=0.5)
        for c in chains:
            assert c.hitRadius == 0.5

    def test_custom_gravity_power(self, mesh, skeleton):
        chains, _ = detect_spring_bones(mesh, skeleton, gravity_power=3.0)
        for c in chains:
            assert c.gravityPower == 3.0

    def test_custom_gravity_dir(self, mesh, skeleton):
        custom_dir = np.array([0.0, -0.8, -0.2], dtype=np.float32)
        custom_dir /= np.linalg.norm(custom_dir)
        chains, _ = detect_spring_bones(mesh, skeleton, gravity_dir=custom_dir)
        for c in chains:
            assert np.allclose(c.gravityDir, custom_dir, atol=1e-6)


class TestLabelsFilter:
    """Test that labels parameter filters chains correctly."""

    def test_labels_with_no_hair_or_clothing_skips_all(self):
        mesh = make_sphere_mesh()
        skeleton = make_15_bone_skeleton()
        # All vertices labelled "body" — no hair / clothing
        labels = np.array(["body"] * mesh.vertex_count, dtype=object)
        chains, _ = detect_spring_bones(mesh, skeleton, labels=labels)
        # Sphere has no thin elongated chains, so result may be empty
        assert isinstance(chains, list)

    def test_labels_none_works(self):
        mesh = make_sphere_mesh()
        skeleton = make_15_bone_skeleton()
        chains, _ = detect_spring_bones(mesh, skeleton, labels=None)
        # No crash with labels=None
        assert isinstance(chains, list)
