"""
Stage 2: Part Segmentation
Labels each vertex with a semantic body part (body, head, hair, clothing, etc.).
Pure functions, full type hints.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.cluster import KMeans

from vtuber_rigger.interfaces import LABEL_SET, LabeledMesh, MeshData


# ─── Pure helpers ─────────────────────────────────────────────────────────────

def _build_adjacency(faces: np.ndarray, n_vertices: int) -> csr_matrix:
    """Build sparse adjacency matrix from triangle faces (Laplacian-style)."""
    rows, cols = [], []
    for tri in faces:
        rows.extend([tri[0], tri[1], tri[2]])
        cols.extend([tri[1], tri[2], tri[0]])
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    return adj + adj.T  # symmetrise


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals via face normals (pure numpy)."""
    n = vertices.shape[0]
    face_norms = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    norms = np.linalg.norm(face_norms, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    face_norms = face_norms / norms

    normals = np.zeros_like(vertices)
    for i, tri in enumerate(faces):
        for vi in tri:
            normals[vi] += face_norms[i]

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0, 1.0, lengths)
    return normals / lengths


def _discrete_laplacian(vertices: np.ndarray, adj: csr_matrix) -> np.ndarray:
    """
    Compute discrete Laplacian curvature per vertex.
    L(v_i) = sum_{j∈N(i)} (v_j - v_i) / |N(i)|
    Returns mean edge deviation magnitude as curvature scalar.
    """
    n = vertices.shape[0]
    # vertex-wise difference to neighbours
    delta = -adj.dot(vertices)  # (n, 3) each row = -sum of neighbour deltas
    # normalise by degree
    degrees = np.array(adj.sum(axis=1)).ravel()
    degrees = np.where(degrees == 0, 1.0, degrees)
    curvature = np.linalg.norm(delta / degrees[:, None], axis=1)
    return curvature


def _elongation_score(cluster_vertices: np.ndarray) -> float:
    """How elongated a cluster is: ratio of bounding box extents."""
    if len(cluster_vertices) < 2:
        return 0.0
    extents = cluster_vertices.max(axis=0) - cluster_vertices.min(axis=0)
    return float(extents.max() / (extents.min() + 1e-9))


def _cluster_to_label(
    cluster_id: int,
    cluster_vertices: np.ndarray,
    cluster_centre_y: float,
    is_attached_to_head: bool,
    is_top_by_y: bool,
    is_largest: bool,
) -> str:
    """Map a KMeans cluster to a semantic label using heuristics."""
    elongation = _elongation_score(cluster_vertices)
    n_verts = len(cluster_vertices)

    # Head: topmost cluster by Y, roughly spherical (low elongation)
    if is_top_by_y and elongation < 2.5:
        return "head"

    # Hair: thin/elongated AND attached to head region
    if elongation > 2.0 and is_attached_to_head:
        return "hair"

    # Body: largest cluster
    if is_largest:
        return "body"

    # Hands/feet: small, far from body centre by Y
    if n_verts < 500:
        return "unknown"

    # Clothing vs accessory: medium size, moderate curvature variance
    return "clothing"


# ─── Public API ───────────────────────────────────────────────────────────────

def segment(mesh: MeshData) -> LabeledMesh:
    """
    Label every vertex of `mesh` with a semantic part label.

    Algorithm
    ---------
    1. Compute vertex normals + discrete Laplacian curvature.
    2. Cluster vertices by (curvature, y_position, normal) with KMeans k∈[6,10].
    3. Heuristic cluster→label mapping (head=topmost, body=largest, hair=elongated+attached).
    4. Connectivity refinement via scipy.sparse.csgraph.

    Parameters
    ----------
    mesh : MeshData
        Normalized triangle mesh from Stage 1.

    Returns
    -------
    LabeledMesh
        Same mesh with an added ``labels`` array of shape (N,) with values in LABEL_SET.

    Raises
    ------
    ValueError
        If the mesh cannot be segmented (e.g. too few vertices).
    """
    if mesh.vertex_count < 20:
        raise ValueError(f"Mesh too small for segmentation: {mesh.vertex_count} vertices")

    verts = mesh.vertices
    faces = mesh.faces
    n = mesh.vertex_count

    # 1. Geometry primitives
    normals = _vertex_normals(verts, faces)
    adj = _build_adjacency(faces, n)
    curvature = _discrete_laplacian(verts, adj)

    # 2. Feature matrix for clustering: (curvature, y_position, normal)
    y_pos = verts[:, 1:2]
    y_range = max(y_pos.max() - y_pos.min(), 1e-9)
    feats = np.hstack([
        curvature.reshape(-1, 1) / (curvature.max() + 1e-9),
        y_pos / y_range,
        normals * 0.5,
    ]).astype(np.float32)

    # 3. KMeans over k ∈ [6, 10], pick best by silhouette score
    from sklearn.metrics import silhouette_score

    best_k, best_score, best_labels = 6, -1.0, None
    for k in range(6, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(feats)
        if k < n:
            try:
                score = silhouette_score(feats, labels, sample_size=min(5000, n))
            except Exception:
                score = -1.0
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int32)

    # 4. Assign semantic labels to clusters
    # Compute per-cluster stats
    cluster_ids = np.unique(best_labels)
    cluster_stats: dict[int, dict] = {}
    for cid in cluster_ids:
        mask = best_labels == cid
        cluster_verts = verts[mask]
        cluster_stats[cid] = {
            "centre_y": cluster_verts[:, 1].mean(),
            "extent_y": cluster_verts[:, 1].max() - cluster_verts[:, 1].min(),
            "n_verts": mask.sum(),
            "elongation": _elongation_score(cluster_verts),
        }

    # Determine reference values
    top_y = max(s["centre_y"] for s in cluster_stats.values())
    largest_cid = max(cluster_stats, key=lambda c: cluster_stats[c]["n_verts"])

    # Build a quick head-vertex set for hair attachment check
    head_cid = None
    for cid, st in cluster_stats.items():
        if st["centre_y"] >= top_y - 0.05 and st["elongation"] < 2.5:
            head_cid = cid
            break
    head_vertices = set(np.where(best_labels == head_cid)[0]) if head_cid is not None else set()

    # Map cluster → label
    semantic_labels = np.empty(n, dtype=object)
    for cid in cluster_ids:
        mask = best_labels == cid
        st = cluster_stats[cid]
        attached = bool(head_vertices & set(np.where(mask)[0])) if head_vertices else False
        label = _cluster_to_label(
            cluster_id=cid,
            cluster_vertices=verts[mask],
            cluster_centre_y=st["centre_y"],
            is_attached_to_head=attached,
            is_top_by_y=(st["centre_y"] >= top_y - 0.05),
            is_largest=(cid == largest_cid),
        )
        semantic_labels[mask] = label

    # 5. Connectivity refinement: ensure "body" is a single connected component
    body_mask = semantic_labels == "body"
    if body_mask.sum() > 0:
        body_adj = adj.tocsr()[body_mask][:, body_mask]
        n_comp, _ = connected_components(body_adj, directed=False)
        if n_comp > 1:
            # Keep largest component as body, relabel rest as clothing
            comp_sizes: dict[int, int] = {}
            for comp_id in range(n_comp):
                comp_sizes[comp_id] = (component_ids == comp_id).sum()
            largest_comp = max(comp_sizes, key=comp_sizes.get)
            component_ids = connected_components(body_adj, directed=False)[1]
            other_body = np.where(
                body_mask & (component_ids != largest_comp)
            )[0]
            for vidx in other_body:
                semantic_labels[vidx] = "clothing"

    # 6. Guarantee every vertex is labeled from LABEL_SET
    for i in range(n):
        if semantic_labels[i] not in LABEL_SET:
            semantic_labels[i] = "unknown"

    result = LabeledMesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        normals=mesh.normals,
        uvs=mesh.uvs,
        material_indices=mesh.material_indices,
        vertex_colors=mesh.vertex_colors,
        labels=np.asarray(semantic_labels),
    )

    # 7. Validate invariants M2-1 .. M2-6
    _validate_invariants(mesh, result)

    return result


# ─── Invariant validators ────────────────────────────────────────────────────

def _validate_invariants(mesh: MeshData, labeled: LabeledMesh) -> None:
    """Raise AssertionError if any M2 invariant is violated."""
    labels = labeled.labels
    n = labeled.vertex_count

    # M2-1: every vertex has exactly one label
    assert labels.shape == (n,), f"M2-1: shape mismatch {labels.shape} vs ({n},)"

    # M2-2: all labels in LABEL_SET
    unseen = {str(l) for l in labels} - LABEL_SET
    assert not unseen, f"M2-2: invalid labels {unseen}"

    # M2-3: label count finite and ≤ |label_set|
    unique = set(str(l) for l in labels)
    assert len(unique) <= len(LABEL_SET), f"M2-3: too many unique labels {len(unique)}"

    # M2-4: body vertices form a single connected component
    body_mask = labels == "body"
    if body_mask.sum() > 0:
        adj = _build_adjacency(mesh.faces, n)
        body_adj = adj.tocsr()[body_mask][:, body_mask]
        n_comp, _ = connected_components(body_adj, directed=False)
        assert n_comp == 1, f"M2-4: body has {n_comp} components, expected 1"

    # M2-5: head vertices are topmost connected region by Y
    head_mask = labels == "head"
    if head_mask.sum() > 0:
        head_y = mesh.vertices[head_mask][:, 1].max()
        body_mask2 = labels == "body"
        if body_mask2.sum() > 0:
            body_y_max = mesh.vertices[body_mask2][:, 1].max()
            assert head_y >= body_y_max, (
                f"M2-5: head max Y {head_y} not above body max Y {body_y_max}"
            )

    # M2-6: hair vertices are connected to head vertices
    hair_mask = labels == "hair"
    if hair_mask.sum() > 0 and head_mask.sum() > 0:
        adj = _build_adjacency(mesh.faces, n)
        hair_idx = np.where(hair_mask)[0]
        head_idx = set(np.where(head_mask)[0])
        # BFS from a hair vertex
        visited = set()
        queue = list(hair_idx[:1].tolist())
        while queue:
            v = queue.pop()
            if v in visited:
                continue
            visited.add(v)
            if v in head_idx:
                break
            # neighbours
            nbrs = adj[v].indices
            for nb in nbrs:
                if nb not in visited:
                    queue.append(nb)
        assert visited & head_idx, "M2-6: hair not connected to head"
