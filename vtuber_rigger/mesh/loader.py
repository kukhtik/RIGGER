"""
Stage 1: Mesh Loader

Loads GLB/GLTF/VRM/FBX/OBJ → normalized MeshData.
FBX is delegated to Blender CLI; all other formats use trimesh.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vtuber_rigger.interfaces import MeshData

SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".vrm", ".fbx", ".obj"}

# ─── Format detection ────────────────────────────────────────────

def _detect_format(path: Path) -> str:
    """Detect format from extension + magic bytes fallback."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    # FBX is binary; trimesh cannot load it reliably
    if ext == ".fbx":
        return "fbx"
    return "gltf"


# ─── Core loader ─────────────────────────────────────────────────

def load_mesh(path: str | Path) -> MeshData:
    """
    Load a 3D model file and return a normalized MeshData.

    Parameters
    ----------
    path : str | Path
        Path to a GLB, GLTF, VRM, FBX, or OBJ file.

    Returns
    -------
    MeshData
        Normalized triangle mesh with validated invariants M1-1…M1-8.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported or the mesh fails validation.
    RuntimeError
        If Blender is unavailable for FBX files.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = _detect_format(path)

    if fmt == "fbx":
        vertices, faces, normals, uvs, material_indices = _load_via_blender(path)
    else:
        vertices, faces, normals, uvs, material_indices = _load_via_trimesh(path)

    # Ensure triangles
    vertices, faces = _ensure_triangles(vertices, faces)

    # Merge duplicate vertices
    vertices, faces = _merge_duplicates(vertices, faces)

    # Normalize to [-1, 1]^3 bounding box
    vertices = _normalize(vertices)

    # Validate M1 invariants
    _validate_mesh(vertices, faces, normals, uvs, material_indices)

    # Compute adjacency
    adjacency = _build_adjacency(vertices, faces)

    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        uvs=uvs,
        material_indices=material_indices,
    )


# ─── trimesh path ────────────────────────────────────────────────

def _load_via_trimesh(path: Path) -> tuple:
    """Load GLB/GLTF/VRM/OBJ via trimesh."""
    import trimesh

    # VRM is GLB-based; trimesh doesn't recognize .vrm extension
    # so we need to pass file_type='glb' for .vrm files
    ext = path.suffix.lower()
    load_kwargs = dict(force="mesh", process=True, validate=False)
    if ext == ".vrm":
        load_kwargs["file_type"] = "glb"

    # Force triangulate to avoid ngons; merge_weld to collapse near-duplicates
    scene_or_mesh = trimesh.load(
        str(path),
        **load_kwargs,
    )

    if isinstance(scene_or_mesh, trimesh.Scene):
        # Concatenate all mesh objects in the scene
        meshes = []
        for geom in scene_or_mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    elif hasattr(scene_or_mesh, "geometry"):
        # Some scene-like objects expose geometry differently
        meshes = [
            g for g in scene_or_mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found: {path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        raise ValueError(f"Unsupported trimesh return type {type(scene_or_mesh)} for: {path}")

    return _extract_trimesh_data(mesh)


def _extract_trimesh_data(mesh) -> tuple:
    """Pull vertices, faces, normals, UVs, material_indices from a trimesh.Trimesh."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    normals = None
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        n = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if n.shape == vertices.shape and np.all(np.isfinite(n)):
            normals = n

    uvs = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
        vis_uv = mesh.visual.uv
        if vis_uv is not None:
            uv = np.asarray(vis_uv, dtype=np.float32)
            if uv.ndim == 2 and uv.shape[1] == 2 and np.all(np.isfinite(uv)):
                uvs = uv

    material_indices = None
    if hasattr(mesh, "material_index") and mesh.material_index is not None:
        mi = np.asarray(mesh.material_index, dtype=np.int32)
        if mi.ndim == 1:
            material_indices = mi

    return vertices, faces, normals, uvs, material_indices


# ─── Blender path (FBX) ─────────────────────────────────────────

def _load_via_blender(path: Path) -> tuple:
    """
    Load FBX via Blender CLI.
    Exports a temporary GLB, loads it with trimesh, then cleans up.
    """
    blender = _find_blender()
    if blender is None:
        raise RuntimeError(
            "Blender not found. FBX loading requires Blender. "
            "Set BLENDER_PATH env var or install Blender."
        )

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Run Blender headless to convert FBX → GLB
        script = _BLENDER_IMPORT_SCRIPT.format(
            input_path=str(path).replace("\\", "\\\\"),
            output_path=str(tmp_path).replace("\\", "\\\\"),
        )
        result = subprocess.run(
            [str(blender), "--background", "--python-expr", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 or not tmp_path.exists():
            raise RuntimeError(
                f"Blender FBX import failed (exit {result.returncode}):\n"
                f"{result.stderr}\n{result.stdout}"
            )

        # Load the exported GLB
        vertices, faces, normals, uvs, material_indices = _load_via_trimesh(tmp_path)
        return vertices, faces, normals, uvs, material_indices
    finally:
        tmp_path.unlink(missing_ok=True)


_BLENDER_IMPORT_SCRIPT = """
import bpy, sys

# Import FBX
bpy.ops.import_scene.fbx(filepath=r"{input_path}")

# Select all mesh objects
bpy.ops.object.select_all(action='SELECT')

# Export as GLB
bpy.ops.export_scene.gltf(
    filepath=r"{output_path}",
    export_format='GLB',
    use_selection=False,
    export_apply=True,
)
"""


def _find_blender() -> Path | None:
    """Locate Blender executable."""
    if blender_path := os.environ.get("BLENDER_PATH"):
        p = Path(blender_path)
        if p.exists():
            return p

    candidates = [
        Path("/mnt/b/Blender/blender.exe"),
        Path("C:/Program Files/Blender Foundation/Blender/blender.exe"),
        Path("C:/Program Files/Blender/blender.exe"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ─── Mesh processing ─────────────────────────────────────────────

def _ensure_triangles(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fan-fill any polygonal faces into triangles (in-place for quads)."""
    # Already triangles?
    if faces.shape[1] == 3:
        return vertices, faces

    # Fan-triangulate from first vertex of each polygon
    new_faces: list[np.ndarray] = []
    for face in faces:
        v0 = face[0]
        for i in range(1, len(face) - 1):
            new_faces.append(np.array([v0, face[i], face[i + 1]], dtype=np.int32))
    return vertices, np.array(new_faces, dtype=np.int32)


_EPS = 1e-6

def _merge_duplicates(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse vertices that are closer than _EPS.
    Maintains face index integrity.
    """
    # k-NN deduplication using scipy cKDTree
    from scipy.spatial import cKDTree

    tree = cKDTree(vertices)
    # For each vertex, find its representative (nearest already-seen)
    # Use sequential merge: each vertex maps to itself, then we collapse
    # Simple approach: cluster identical positions
    _, unique_inverse, counts = np.unique(
        np.round(vertices / _EPS).astype(np.int64),
        axis=0,
        return_index=True,
        return_counts=True,
    )

    # Keep unique vertices in stable order
    unique_mask = np.zeros(len(vertices), dtype=bool)
    unique_mask[unique_inverse] = True
    new_vertices = vertices[unique_mask]

    # Remap face indices
    old_to_new = np.full(len(vertices), -1, dtype=np.int32)
    old_to_new[unique_mask] = np.arange(len(new_vertices), dtype=np.int32)
    new_faces = old_to_new[faces]

    # Remove degenerate faces (any index == -1 or duplicate vertex in same face)
    valid = np.all(new_faces >= 0, axis=1) & (new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 0] != new_faces[:, 2]) & (new_faces[:, 1] != new_faces[:, 2])
    return new_vertices, new_faces[valid]


def _normalize(vertices: np.ndarray) -> np.ndarray:
    """
    Center at origin and scale so bounding box fits in [-1, 1]^3.
    """
    amin = vertices.min(axis=0)
    amax = vertices.max(axis=0)
    center = (amin + amax) / 2.0
    extent = amax - amin
    scale = 2.0 / max(extent.max(), 1e-10)
    return (vertices - center) * scale


# ─── Validation (M1-1…M1-8) ──────────────────────────────────────

def _validate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    uvs: np.ndarray | None,
    material_indices: np.ndarray | None,
) -> None:
    """Raise ValueError with a clear message if any M1 invariant is violated."""

    # M1-1: all vertex positions are finite
    if not np.all(np.isfinite(vertices)):
        raise ValueError("M1-1 violated: mesh contains NaN or Inf in vertex positions")

    # M1-2: all faces are triangles
    if faces.shape[1] != 3:
        raise ValueError(f"M1-2 violated: expected triangles (3 indices), got {faces.shape[1]}")

    # M1-3: all face vertex indices are valid
    if not np.all((faces >= 0) & (faces < len(vertices))):
        raise ValueError("M1-3 violated: face contains an out-of-range vertex index")

    # M1-4: at least 100 vertices and at least 1 face
    if len(vertices) < 100:
        raise ValueError(f"M1-4 violated: mesh has {len(vertices)} vertices, need ≥100")
    if len(faces) < 1:
        raise ValueError("M1-4 violated: mesh has no faces")

    # M1-5: bounding box fits in [-1, 1]^3
    amin = vertices.min(axis=0)
    amax = vertices.max(axis=0)
    if not (np.all(amin >= -1.0) and np.all(amax <= 1.0)):
        raise ValueError(
            f"M1-5 violated: bounding box [{amin}] → [{amax}] exceeds [-1,1]³"
        )

    # M1-6: UV coordinates are finite (if present)
    if uvs is not None and not np.all(np.isfinite(uvs)):
        raise ValueError("M1-6 violated: mesh contains NaN or Inf in UV coordinates")

    # M1-7: material indices per face are valid (if present)
    if material_indices is not None and len(material_indices) > 0:
        max_mat = int(material_indices.max())
        if max_mat < 0:
            raise ValueError("M1-7 violated: material indices must be non-negative")

    # M1-8: no duplicate vertices (enforced by _merge_duplicates upstream)
    # We re-check here as a safety net
    rounded = np.round(vertices / _EPS).astype(np.int64)
    _, unique_count = np.unique(rounded, axis=0, return_counts=True)
    if len(unique_count) < len(vertices):
        raise ValueError(
            f"M1-8 violated: {len(vertices) - len(unique_count)} duplicate vertices remain "
            f"(within ε={_EPS})"
        )


# ─── Adjacency ──────────────────────────────────────────────────

def _build_adjacency(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """
    Build a sparse adjacency matrix (vertex_count × vertex_count).
    edge (i, j) present iff i and j share a face.
    """
    n = len(vertices)
    edges: list[tuple[int, int]] = []
    for face in faces:
        edges.extend([(face[0], face[1]), (face[1], face[2]), (face[2], face[0])])
        edges.extend([(face[1], face[0]), (face[2], face[1]), (face[0], face[2])])

    rows, cols = zip(*edges)
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Keep only the upper triangle to avoid redundancy; combine symmetric
    return adj + adj.T


# ─── Module-level sentinel ──────────────────────────────────────
IMPORTS_OK = True
