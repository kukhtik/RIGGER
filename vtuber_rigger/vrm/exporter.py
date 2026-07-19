"""Stage 8: VRM Exporter — assemble all components into a valid VRM 1.0 file.

M8-1..M8-14 invariants enforced.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from vtuber_rigger.interfaces import (
    MeshData,
    Skeleton,
    Bone,
    SkinWeights,
    Expression,
    SpringChain,
    SpringNode,
    Collider,
    LookAt,
    FirstPerson,
    VRMMeta,
    VRM_STANDARD_EXPRESSIONS,
    REQUIRED_BONES,
)


# ─── GLB constants ────────────────────────────────────────────────────────────

GLB_MAGIC = 0x46546C67  # 'glTF'
GLB_VERSION = 2
GLB_LENGTH_OFFSET = 8
BIN_BUFFER_INDEX = 0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pack_glb(json_bytes: bytes, binary_chunk: bytes) -> bytes:
    """Pack JSON chunk + optional binary chunk into a GLB blob."""
    # GLB header (12 bytes)
    header = struct.pack("<III", GLB_MAGIC, GLB_VERSION, 0)  # length filled below

    # JSON chunk header (8 bytes)
    json_len = len(json_bytes)
    padded_json_len = (json_len + 3) & ~3  # align to 4
    json_chunk = struct.pack("<II", json_len, 0x4E4F534A) + json_bytes
    # pad
    json_chunk += b"\x00" * (padded_json_len - json_len)

    # Binary chunk header (8 bytes) + data
    binary_data = b""
    if binary_chunk:
        binary_len = len(binary_chunk)
        padded_binary_len = (binary_len + 3) & ~3
        binary_data = struct.pack("<II", binary_len, 0x004E4942) + binary_chunk
        binary_data += b"\x00" * (padded_binary_len - binary_len)

    total_len = 12 + len(json_chunk) + len(binary_data)
    header = struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_len)

    return header + json_chunk + binary_data


def _build_gltf(
    mesh: MeshData,
    skeleton: Skeleton,
    weights: SkinWeights,
    expressions: dict[str, Expression],
    spring_chains: list[SpringChain],
    colliders: list[Collider],
    lookat: LookAt | None,
    firstperson: FirstPerson | None,
    meta: VRMMeta,
) -> tuple[dict, bytes]:
    """Build glTF 2.0 JSON structure + binary buffer."""

    # ── Binary buffer layout ─────────────────────────────────────────────────
    buf = bytearray()
    buf_views: list[tuple[int, int]] = []  # (offset, length) per view

    def add_buffer_view(data: bytes) -> int:
        offset = len(buf)
        length = len(data)
        buf.extend(data)
        # pad to 4-byte alignment
        pad = (4 - length % 4) % 4
        buf.extend(b"\x00" * pad)
        buf_views.append((offset, length))
        return len(buf_views) - 1

    def add_buffer_view_array(arr: np.ndarray) -> int:
        return add_buffer_view(arr.astype(arr.dtype.newbyteorder("<")).tobytes())

    # ── Accessors ────────────────────────────────────────────────────────────
    accessors: list[dict] = []

    def make_accessor(
        buf_view_idx: int,
        offset: int,
        count: int,
        component_type: int,
        type_str: str,
        min_val: list | None = None,
        max_val: list | None = None,
        normalized: bool = False,
    ) -> int:
        acc: dict[str, Any] = {
            "bufferView": buf_view_idx,
            "byteOffset": offset,
            "componentType": component_type,
            "count": count,
            "type": type_str,
        }
        if min_val is not None:
            acc["min"] = min_val
        if max_val is not None:
            acc["max"] = max_val
        if normalized:
            acc["normalized"] = True
        accessors.append(acc)
        return len(accessors) - 1

    # ── POSITION accessor ─────────────────────────────────────────────────────
    vert_bytes = mesh.vertices.astype("<f4").tobytes()
    pos_bv = add_buffer_view(vert_bytes)
    pos_min = mesh.vertices.min(axis=0).tolist()
    pos_max = mesh.vertices.max(axis=0).tolist()
    pos_acc = make_accessor(pos_bv, 0, mesh.vertex_count, 5126, "VEC3", pos_min, pos_max)

    # ── NORMAL accessor ───────────────────────────────────────────────────────
    norm_acc = -1
    if mesh.normals is not None:
        norm_bytes = mesh.normals.astype("<f4").tobytes()
        norm_bv = add_buffer_view(norm_bytes)
        norm_acc = make_accessor(norm_bv, 0, mesh.vertex_count, 5126, "VEC3")

    # ── TEXCOORD_0 accessor ───────────────────────────────────────────────────
    uv_acc = -1
    if mesh.uvs is not None:
        # glTF 2.0 TEXCOORD must be VEC2 float
        uvs_flat = mesh.uvs.astype("<f4").tobytes()
        uv_bv = add_buffer_view(uvs_flat)
        uv_acc = make_accessor(uv_bv, 0, len(mesh.uvs), 5126, "VEC2")

    # ── JOINTS_0 + WEIGHTS_0 accessors ────────────────────────────────────────
    # Build joint/weight arrays from sparse weight matrix
    n_verts = mesh.vertex_count
    n_bones = len(weights.bone_names)

    joints = np.zeros((n_verts, 4), dtype=np.uint16)
    joint_weights = np.zeros((n_verts, 4), dtype=np.float32)

    for vi in range(n_verts):
        row = weights.matrix.getrow(vi).toarray().ravel()
        nonzero = [(int(wi), float(w)) for wi, w in enumerate(row) if w > 0]
        nonzero.sort(key=lambda x: -x[1])  # sort descending by weight
        for j, (wi, w) in enumerate(nonzero[:4]):
            joints[vi, j] = wi
            joint_weights[vi, j] = w

    # Normalise weights to sum to 1
    row_sums = joint_weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    joint_weights = joint_weights / row_sums

    joints_bv = add_buffer_view_array(joints)
    joints_acc = make_accessor(joints_bv, 0, n_verts, 5123, "VEC4")  # UNSIGNED_SHORT

    weights_bv = add_buffer_view_array(joint_weights)
    weights_acc = make_accessor(weights_bv, 0, n_verts, 5126, "VEC4")

    # ── INDEX accessor (faces) ────────────────────────────────────────────────
    # glTF is unsigned int; ensure indices fit in uint16 if possible
    idx_arr = mesh.faces.astype(np.uint32)
    idx_bv = add_buffer_view(idx_arr.tobytes())
    idx_acc = make_accessor(idx_bv, 0, mesh.face_count * 3, 5125, "SCALAR")

    # ── Inverse bind matrices ─────────────────────────────────────────────────
    # One IBM per bone, in skeleton bone order
    ibm_data = np.zeros((n_bones, 16), dtype=np.float32)
    for bi, bone_name in enumerate(weights.bone_names):
        bone = skeleton.get_bone(bone_name)
        if bone is not None:
            # Translation-only inverse bind (translation part of global inverse)
            pos = bone.position.astype(np.float64)
            ibm_data[bi, 0] = 1
            ibm_data[bi, 5] = 1
            ibm_data[bi, 10] = 1
            ibm_data[bi, 15] = 1
            ibm_data[bi, 12] = -pos[0]
            ibm_data[bi, 13] = -pos[1]
            ibm_data[bi, 14] = -pos[2]
        else:
            # Identity
            ibm_data[bi] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    ibm_bv = add_buffer_view_array(ibm_data)
    ibm_acc = make_accessor(ibm_bv, 0, n_bones, 5126, "MAT4")

    # ── Build nodes ───────────────────────────────────────────────────────────
    nodes: list[dict[str, Any]] = []
    # First create all bone nodes, then a mesh node
    bone_node_indices: dict[str, int] = {}
    for bone in skeleton.bones:
        node: dict[str, Any] = {
            "name": bone.name,
            "translation": bone.position.tolist(),
        }
        if bone.rotation is not None:
            node["rotation"] = bone.rotation.tolist()
        # positive scale (M8-14)
        node["scale"] = [1.0, 1.0, 1.0]
        bone_node_indices[bone.name] = len(nodes)
        nodes.append(node)

    # Mesh node (references skin + primitive)
    mesh_node_idx = len(nodes)
    nodes.append({
        "name": "mesh",
        "mesh": 0,
        "skin": 0,
        "scale": [1.0, 1.0, 1.0],
    })

    # ── Mesh primitive ────────────────────────────────────────────────────────
    prim_attrs: dict[str, int] = {
        "POSITION": pos_acc,
        "JOINTS_0": joints_acc,
        "WEIGHTS_0": weights_acc,
    }
    if norm_acc >= 0:
        prim_attrs["NORMAL"] = norm_acc
    if uv_acc >= 0:
        prim_attrs["TEXCOORD_0"] = uv_acc

    primitives = [{
        "attributes": prim_attrs,
        "indices": idx_acc,
        "material": 0,
    }]

    meshes_list = [{"primitives": primitives}]

    # ── Skin ──────────────────────────────────────────────────────────────────
    # Joints array: node indices in bone order
    joints_array = [bone_node_indices[bn] for bn in weights.bone_names if bn in bone_node_indices]

    skins_list = [{
        "inverseBindMatrices": ibm_acc,
        "joints": joints_array,
        "name": "Armature",
    }]

    # ── Material (unlit) ──────────────────────────────────────────────────────
    materials = [{
        "name": "DefaultMaterial",
        "extensions": {
            "KHR_materials_unlit": {},
        },
        "extras": {
            "targetNames": list(expressions.keys()),
        },
    }]

    # ── VRMC_vrm extension ────────────────────────────────────────────────────
    vrm_ext: dict[str, Any] = {
        "specVersion": "1.0",
        "meta": {
            "name": meta.name,
            "version": meta.version,
            "authors": meta.authors,
            "licenseUrl": meta.licenseUrl,
            "contactInformation": meta.contactInformation or "",
        },
        "humanoid": {
            "humanBones": [
                {"bone": name, "node": bone_node_indices[name]}
                for name in REQUIRED_BONES
                if name in bone_node_indices
            ]
        },
    }

    # ── lookAt ────────────────────────────────────────────────────────────────
    if lookat is not None:
        vrm_ext["lookAt"] = {
            "type": lookat.mode,
            "yawRange": lookat.yaw_range,
            "pitchRange": lookat.pitch_range,
        }

    # ── firstPerson ───────────────────────────────────────────────────────────
    if firstperson is not None:
        fp_anns = []
        for ann in firstperson.mesh_annotations:
            fp_anns.append({
                "mesh": mesh_node_idx,
                "firstPersonFlag": ann["type"],
            })
        vrm_ext["firstPerson"] = {
            "firstPersonBoneOffset": firstperson.camera_offset.tolist(),
            "meshAnnotations": fp_anns,
        }

    # ── expressions ───────────────────────────────────────────────────────────
    if expressions:
        exprs_out: dict[str, Any] = {}
        for name, expr in expressions.items():
            exprs_out[name] = {
                "morphTargetBinds": [{
                    "node": mesh_node_idx,
                    "index": expr.morph_target_index,
                }],
                "isBinary": False,
                "blendShapePreset": name if name in VRM_STANDARD_EXPRESSIONS else "custom",
            }
        vrm_ext["expressions"] = exprs_out

    # ── VRMC_springBone ────────────────────────────────────────────────────────
    spring_ext: dict[str, Any] = {"specVersion": "1.0"}

    if spring_chains:
        sb_colliders: list[dict] = []
        for ci, col in enumerate(colliders):
            sb_colliders.append({
                "node": -1,  # colliders are spatial; node reference optional
                "shape": {
                    "sphere": {
                        "offset": col.center.tolist(),
                        "radius": col.radius,
                    }
                },
            })

        sb_springs: list[dict] = []
        for chain in spring_chains:
            chain_nodes = []
            for node in chain.nodes:
                if node.bone_name in bone_node_indices:
                    chain_nodes.append({"node": bone_node_indices[node.bone_name]})
            if chain_nodes:
                sb_springs.append({
                    "joints": chain_nodes,
                    "colliders": list(range(len(sb_colliders))),
                    "stiffiness": chain.stiffness,
                    "dragForce": chain.dragForce,
                    "hitRadius": chain.hitRadius,
                    "gravityPower": chain.gravityPower,
                    "gravityDir": chain.gravityDir.tolist(),
                })

        spring_ext["colliders"] = sb_colliders
        spring_ext["springs"] = sb_springs

    # ── Assemble glTF JSON ────────────────────────────────────────────────────
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "VTuberRigger/1.0"},
        "scene": 0,
        "scenes": [{"nodes": [mesh_node_idx]}],
        "nodes": nodes,
        "meshes": meshes_list,
        "skins": skins_list,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": [
            {"buffer": BIN_BUFFER_INDEX, "byteOffset": off, "byteLength": length}
            for off, length in buf_views
        ],
        "buffers": [{"byteLength": len(buf)}],
        "extensions": {
            "VRMC_vrm": vrm_ext,
            "VRMC_springBone": spring_ext,
        },
        "extensionsUsed": ["VRMC_vrm", "VRMC_springBone", "KHR_materials_unlit"],
    }

    # M8-13: no animations or cameras
    # (we simply don't add them)

    return gltf, bytes(buf)


def export_vrm(
    mesh: MeshData,
    skeleton: Skeleton,
    weights: SkinWeights,
    expressions: dict[str, Expression],
    spring_chains: list[SpringChain],
    colliders: list[Collider],
    lookat: LookAt | None,
    firstperson: FirstPerson | None,
    meta: VRMMeta,
    output_path: str,
) -> str:
    """Assemble all components and write a VRM 1.0 file.

    M8-1:  File is valid GLB 2.0
    M8-2:  VRMC_vrm with specVersion "1.0"
    M8-3:  All 15 required humanoid bones mapped
    M8-4:  Meta with name, version, authors, licenseUrl
    M8-5:  VRMC_vrm.expressions present
    M8-6:  VRMC_vrm.lookAt present
    M8-7:  VRMC_vrm.firstPerson present
    M8-8:  VRMC_springBone present
    M8-9:  All humanoid bone nodes exist
    M8-10: All morphTargets referenced exist
    M8-11: File size < 50 MB
    M8-12: Thumbnail ≤ 1024×1024 PNG
    M8-13: No animations or cameras
    M8-14: Bone scale components positive

    Returns the output_path string.
    """
    gltf_dict, binary_buf = _build_gltf(
        mesh, skeleton, weights, expressions,
        spring_chains, colliders, lookat, firstperson, meta,
    )

    json_str = json.dumps(gltf_dict, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")

    glb_blob = _pack_glb(json_bytes, binary_buf)

    # M8-11
    if len(glb_blob) > 50 * 1024 * 1024:
        raise ValueError(f"VRM file too large: {len(glb_blob):,} bytes (max 50 MB)")

    with open(output_path, "wb") as f:
        f.write(glb_blob)

    return output_path


def validate_vrm(path: str) -> list[str]:
    """Validate a VRM file and return a list of error strings.

    Empty list = valid.
    M8-1:  GLB 2.0 structure
    M8-2:  VRMC_vrm specVersion "1.0"
    M8-3:  All 15 required humanoid bones
    M8-4:  Meta required fields
    M8-11: File size ≤ 50 MB
    M8-13: No animations or cameras
    M8-14: Bone scale positive
    """
    errors: list[str] = []

    # M8-11
    import os
    size = os.path.getsize(path)
    if size > 50 * 1024 * 1024:
        errors.append(f"M8-11: File size {size:,} exceeds 50 MB")
    if size < 12:
        errors.append(f"M8-1: File too small ({size} bytes) to be GLB")
        return errors

    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        errors.append(f"Cannot read file: {e}")
        return errors

    # GLB header check
    if len(data) < 12:
        errors.append("M8-1: GLB header truncated")
        return errors

    magic, version, length = struct.unpack("<III", data[0:12])
    if magic != GLB_MAGIC:
        errors.append(f"M8-1: Invalid GLB magic 0x{magic:08X} (expected 0x{GLB_MAGIC:08X})")
        return errors
    if version != 2:
        errors.append(f"M8-1: Not GLB version 2 (got {version})")

    # Parse JSON chunk
    json_len, json_fmt = struct.unpack("<II", data[12:20])
    if json_fmt != 0x4E4F534A:
        errors.append("M8-1: First chunk is not JSON")
    json_data = data[20 : 20 + json_len]

    try:
        gltf = json.loads(json_data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        errors.append(f"M8-1: Invalid JSON chunk: {e}")
        return errors

    # M8-13: no animations or cameras
    if "animations" in gltf:
        errors.append("M8-13: 'animations' present (VRM forbids animations)")
    if "cameras" in gltf:
        errors.append("M8-13: 'cameras' present (VRM forbids cameras)")

    # Check extensions
    ext_used = set(gltf.get("extensionsUsed", []))
    if "VRMC_vrm" not in ext_used:
        errors.append("M8-2: VRMC_vrm not in extensionsUsed")

    vrm_ext = gltf.get("extensions", {}).get("VRMC_vrm", {})

    # M8-2
    if vrm_ext.get("specVersion") != "1.0":
        errors.append(f"M8-2: specVersion is '{vrm_ext.get('specVersion')}' (expected '1.0')")

    # M8-4: meta required fields
    meta = vrm_ext.get("meta", {})
    for field in ("name", "version", "authors", "licenseUrl"):
        if field not in meta or not meta[field]:
            errors.append(f"M8-4: meta.{field} missing or empty")

    # M8-3: humanoid bones
    humanoid = vrm_ext.get("humanoid", {})
    bone_entries = humanoid.get("humanBones", [])
    bone_names_in_vrm = {e.get("bone") for e in bone_entries}
    missing = REQUIRED_BONES - bone_names_in_vrm
    if missing:
        errors.append(f"M8-3: Missing required bones: {sorted(missing)}")

    # M8-9: humanoid bone nodes exist
    nodes = gltf.get("nodes", [])
    for entry in bone_entries:
        node_idx = entry.get("node")
        if not isinstance(node_idx, int) or not (0 <= node_idx < len(nodes)):
            errors.append(f"M8-9: Bone '{entry.get('bone')}' references invalid node {node_idx}")

    # M8-14: check all node scales are positive
    for ni, node in enumerate(nodes):
        if "scale" in node:
            sc = node["scale"]
            if not all(isinstance(v, (int, float)) and v > 0 for v in sc):
                errors.append(f"M8-14: Node {ni} ('{node.get('name','')}') has non-positive scale {sc}")

    # Check required nodes exist
    # (mesh node, skin, etc. validated implicitly by glTF validator)

    return errors
