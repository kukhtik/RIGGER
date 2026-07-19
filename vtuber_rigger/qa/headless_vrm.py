from __future__ import annotations
import json
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
from pygltflib import GLTF2
from typing import Any, Optional
from dataclasses import dataclass

from vtuber_rigger.interfaces import (
    MeshData,
    Skeleton,
    Bone,
    SkinWeights,
    Expression,
    MorphTarget,
    SpringChain,
    SpringNode,
    Collider,
    LookAt,
    FirstPerson,
    VRMMeta,
    VRM_STANDARD_EXPRESSIONS,
    REQUIRED_BONES,
)
from vtuber_rigger.vrm.exporter import export_vrm as vrm_export_impl

@dataclass
class VRMData:
    mesh: MeshData
    skeleton: Skeleton
    weights: SkinWeights
    expressions: dict[str, Expression]
    spring_chains: list[SpringChain]
    colliders: list[Collider]
    lookat: Optional[LookAt]
    firstperson: Optional[FirstPerson]
    meta: VRMMeta

def load_vrm(path: str) -> VRMData:
    # Parse GLB + VRMC_vrm extensions into VRMData.
    # GLTF2.load handles both .gltf and .glb
    gltf = GLTF2.load(path)
    
    # 1. Extract Binary Buffers
    bin_data = gltf.binary_blob()
    
    def get_accessor_data(acc_idx: int) -> np.ndarray:
        acc = gltf.accessors[acc_idx]
        bv = gltf.bufferViews[acc.bufferView]
        
        # Component types
        type_map = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 
                    5123: np.uint16, 5125: np.uint32, 5126: np.float32}
        dtype = type_map[acc.componentType]
        
        # Type dimensions
        dim_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
        count = acc.count
        dim = dim_map[acc.type]
        
        offset = bv.byteOffset + (acc.byteOffset or 0)
        element_size = np.dtype(dtype).itemsize * dim
        data_bytes = bin_data[offset : offset + count * element_size]
        
        return np.frombuffer(data_bytes, dtype=dtype).reshape(count, dim)

    # 2. Mesh Data
    mesh_idx = next(n.mesh for n in gltf.nodes if n.mesh is not None)
    prim = gltf.meshes[mesh_idx].primitives[0]
    
    # Position, Normal, UV
    verts = get_accessor_data(prim.attributes.POSITION)
    normals = get_accessor_data(prim.attributes.NORMAL) if prim.attributes.NORMAL is not None else None
    uvs = get_accessor_data(prim.attributes.TEXCOORD_0) if prim.attributes.TEXCOORD_0 is not None else None
    faces = get_accessor_data(prim.indices).flatten().astype(np.int32).reshape(-1, 3)
    
    mesh_data = MeshData(vertices=verts, faces=faces, normals=normals, uvs=uvs)

    # 3. Skeleton
    bones = []
    for i, node in enumerate(gltf.nodes):
        if node.translation is not None:
            bones.append(Bone(
                name=node.name or f"node_{i}",
                position=np.array(node.translation, dtype=np.float32),
                rotation=np.array(node.rotation, dtype=np.float32) if node.rotation else None
            ))
    skeleton = Skeleton(bones=bones)

    # 4. Weights
    skin = gltf.skins[0]
    joint_acc = get_accessor_data(skin.joints)
    weight_acc = get_accessor_data(skin.weights)
    
    from scipy.sparse import csr_matrix
    rows = np.repeat(np.arange(len(weight_acc)), 4)
    cols = joint_acc.flatten()
    vals = weight_acc.flatten()
    
    bone_names_in_skin = [gltf.nodes[idx].name if gltf.nodes[idx].name else f"node_{idx}" for idx in skin.joints[0]]
    weights = SkinWeights(matrix=csr_matrix((vals, (rows, cols))), bone_names=bone_names_in_skin)

    # 5. Expressions
    vrm_ext = gltf.extensions.get("VRMC_vrm", {})
    expr_map = {}
    vrm_exprs = vrm_ext.get("expressions", {})
    for name, data in vrm_exprs.items():
        target_idx = data["morphTargetBinds"][0]["index"]
        deltas = get_accessor_data(gltf.meshes[mesh_idx].morphTargets[target_idx])
        abs_verts = verts + deltas
        target = MorphTarget(name=name, vertices=abs_verts, vertex_count=len(abs_verts))
        expr_map[name] = Expression(name=name, morph_target=target, morph_target_index=target_idx)

    # 6. Spring Bones
    spring_ext = gltf.extensions.get("VRMC_springBone", {})
    spring_chains = []
    for s in spring_ext.get("springs", []):
        nodes = []
        for j in s["joints"]:
            node_idx = j["node"]
            node_name = gltf.nodes[node_idx].name
            pos = np.array(gltf.nodes[node_idx].translation or [0,0,0], dtype=np.float32)
            nodes.append(SpringNode(bone_name=node_name, position=pos))
        spring_chains.append(SpringChain(
            nodes=nodes,
            root_bone=nodes[0].bone_name,
            stiffness=s.get("stiffiness", 0.5),
            dragForce=s.get("dragForce", 0.5),
            hitRadius=s.get("hitRadius", 0.02),
            gravityPower=s.get("gravityPower", 0.5),
            gravityDir=np.array(s.get("gravityDir", [0, -1, 0]), dtype=np.float32)
        ))

    colliders = []
    for c in spring_ext.get("colliders", []):
        sphere = c["shape"]["sphere"]
        colliders.append(Collider(
            center=np.array(sphere["offset"], dtype=np.float32),
            radius=sphere["radius"]
        ))

    lookat_data = vrm_ext.get("lookAt")
    lookat = LookAt(mode=lookat_data["type"], yaw_range=lookat_data["yawRange"], pitch_range=lookat_data["pitchRange"]) if lookat_data else None
    
    fp_data = vrm_ext.get("firstPerson")
    firstperson = None
    if fp_data:
        firstperson = FirstPerson(
            camera_offset=np.array(fp_data["firstPersonBoneOffset"], dtype=np.float32),
            mesh_annotations=[{"node": a["mesh"], "type": a["firstPersonFlag"]} for a in fp_data["meshAnnotations"]]
        )

    meta_data = vrm_ext.get("meta", {})
    meta = VRMMeta(
        name=meta_data.get("name", "Unknown"),
        version=meta_data.get("version", "1.0"),
        authors=meta_data.get("authors", ["Unknown"]),
        licenseUrl=meta_data.get("licenseUrl", ""),
        contactInformation=meta_data.get("contactInformation", "")
    )

    return VRMData(mesh_data, skeleton, weights, expr_map, spring_chains, colliders, lookat, firstperson, meta)

def apply_expression(vrm: VRMData, expr_name: str, weight: float = 1.0) -> VRMData:
    # Apply morph target interpolation.
    if expr_name not in vrm.expressions:
        return vrm
    
    expr = vrm.expressions[expr_name]
    target_verts = expr.morph_target.vertices
    base_verts = vrm.mesh.vertices
    
    new_verts = base_verts + weight * (target_verts - base_verts)
    
    new_mesh = MeshData(
        vertices=new_verts,
        faces=vrm.mesh.faces,
        normals=vrm.mesh.normals,
        uvs=vrm.mesh.uvs,
        material_indices=vrm.mesh.material_indices,
        vertex_colors=vrm.mesh.vertex_colors
    )
    
    import copy
    new_vrm = copy.deepcopy(vrm)
    new_vrm.mesh = new_mesh
    return new_vrm

def rotate_bone(vrm: VRMData, bone_name: str, x: float = 0, y: float = 0, z: float = 0) -> VRMData:
    # Rotate bone by degrees, compute deformed mesh via skin weights.
    import copy
    new_vrm = copy.deepcopy(vrm)
    bone = new_vrm.skeleton.get_bone(bone_name)
    if bone is None:
        return vrm
    
    rot = R.from_euler('xyz', [x, y, z], degrees=True).as_matrix()
    
    verts = vrm.mesh.vertices
    deformed_verts = verts.copy()
    
    try:
        bone_idx = vrm.weights.bone_names.index(bone_name)
    except ValueError:
        return vrm
        
    col = vrm.weights.matrix.getcol(bone_idx).toarray().flatten()
    
    joint_pos = bone.position
    
    for i, weight in enumerate(col):
        if weight > 0:
            rel_pos = verts[i] - joint_pos
            rot_pos = rot @ rel_pos
            new_pos = rot_pos + joint_pos
            deformed_verts[i] += weight * (new_pos - verts[i])
            
    new_mesh = MeshData(
        vertices=deformed_verts,
        faces=vrm.mesh.faces,
        normals=vrm.mesh.normals,
        uvs=vrm.mesh.uvs
    )
    new_vrm.mesh = new_mesh
    return new_vrm

def apply_lookat(vrm: VRMData, yaw: float, pitch: float) -> VRMData:
    # Apply eye bone rotation for lookAt.
    vrm = rotate_bone(vrm, "leftEye", x=pitch, y=yaw)
    vrm = rotate_bone(vrm, "rightEye", x=pitch, y=yaw)
    return vrm


def simulate_spring_physics(vrm: "VRMData", impulse: tuple = (1.0, 0, 0),
                            frames: int = 120, fps: int = 60) -> list:
    """Verlet integration for spring bones. Returns list of {bone_name: position} per frame."""
    if not vrm.spring_chains:
        return []

    dt = 1.0 / fps
    results = []

    # Simple Verlet: for each chain, simulate tip node oscillation
    for chain in vrm.spring_chains:
        if len(chain.nodes) < 1:
            continue
        tip_node = chain.nodes[-1]
        pos = tip_node.position.copy().astype(np.float32)
        vel = np.array(impulse, dtype=np.float32)
        stiffness = chain.stiffness
        drag = chain.dragForce
        gravity = chain.gravityPower * chain.gravityDir.astype(np.float32)

        for _ in range(frames):
            # Force = spring (toward rest) + gravity + drag
            rest = tip_node.position.astype(np.float32)
            spring_force = -stiffness * (pos - rest)
            total_force = spring_force + gravity - drag * vel
            vel = vel + total_force * dt
            pos = pos + vel * dt
            results.append({tip_node.bone_name: pos.copy()})

    return results


def export_vrm(vrm: "VRMData", path: str) -> str:
    """Re-export VRM from VRMData. Delegates to vtuber_rigger.vrm.exporter."""
    from vtuber_rigger.vrm.exporter import export_vrm as _export
    return _export(
        mesh=vrm.mesh,
        skeleton=vrm.skeleton,
        weights=vrm.weights,
        expressions=vrm.expressions,
        spring_chains=vrm.spring_chains,
        colliders=vrm.colliders,
        lookat=vrm.lookat,
        firstperson=vrm.firstperson,
        meta=vrm.meta,
        output_path=path,
    )
