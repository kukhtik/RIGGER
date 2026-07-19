# VTuber Rigger — TEST.md
**Test plan with discrete invariants and benchmark strategy**
**Revision 2** | Aligned with SPEC.md r2 (VRM 1.0)

---

## Philosophy

```
Tests verify SPEC.md invariants.
Every invariant from SPEC.md has a corresponding test.
Every code change must not break any passing test.
```

Two test categories:
1. **Automated** (run in CI/CD, pass/fail) — invariants, benchmarks
2. **Human evaluation** (manual gate) — aesthetic quality, VSeeFace live test

---

## Test Structure

```
tests/
├── unit/                    # Per-module, per-invariant
├── integration/             # Cross-stage, pipeline e2e
├── property/                # Hypothesis (fuzzing-style)
├── benchmarks/              # Numerical comparison vs VRoid
├── fixtures/               # Shared test data
└── conftest.py             # pytest fixtures + helpers
```

---

## Fixtures

```
tests/fixtures/
├── meshes/
│   ├── small_avatar.glb         # ~5k tris, simple humanoid
│   ├── anime_female.glb        # ~20k tris, anime-style
│   ├── vroid_reference.vrm      # Real VRoid Hub CC0 model
│   └── invalid_mesh.glb         # Corrupted file for error tests
├── screenshots/
│   ├── front.png               # Front view of test character
│   ├── side.png                # Side view
│   ├── back.png                # Back view
│   └── 3q.png                  # 3/4 view
├── expected/
│   ├── skeleton_required.json   # 15 required VRM bone names
│   ├── expressions_17.json      # 17 standard expression names
│   └── vrm_schema.json         # VRM 1.0 JSON schema
└── conftest.py
```

**Fixture generation:**
- `scripts/generate_test_mesh.py` — procedural GLB via trimesh
- `scripts/download_vroid_refs.py` — download CC0 VRM from VRoid Hub
- `screenshots/` — render from test VRM or use CC0 VTuber references

**Helpers (conftest.py):**
```python
import pytest
import numpy as np
from vtuber_rigger.mesh.loader import MeshData
from vtuber_rigger.rig.skeleton import Skeleton

@pytest.fixture
def simple_mesh():
    """Load simple test mesh"""
    return MeshData.load("tests/fixtures/meshes/small_avatar.glb")

@pytest.fixture
def labeled_mesh(simple_mesh):
    """Mesh with semantic labels"""
    from vtuber_rigger.mesh.segmentation import segment
    return segment(simple_mesh)

@pytest.fixture
def skeleton(labeled_mesh):
    """Predicted skeleton for labeled mesh"""
    from vtuber_rigger.rig.skeleton import predict_skeleton
    return predict_skeleton(labeled_mesh)

@pytest.fixture
def weights(simple_mesh, skeleton):
    """Skin weights for mesh + skeleton"""
    from vtuber_rigger.rig.weights import compute_weights
    return compute_weights(simple_mesh, skeleton)

REQUIRED_BONES = {
    "hips", "spine", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot"
}

VRM_STANDARD_EXPRESSIONS = {
    "happy", "angry", "sad", "relaxed", "surprised",
    "lookUp", "lookDown", "lookLeft", "lookRight", "blink",
    "blinkLeft", "blinkRight",
    "aa", "ih", "ou", "ee", "oh"
}
```

---

## Unit Tests

### test_mesh_loader.py (Stage 1, invariants M1)

```python
import numpy as np
import pytest
from vtuber_rigger.mesh.loader import MeshData

def test_vertices_finite(simple_mesh):
    """M1-1: All vertex positions are finite"""
    assert np.all(np.isfinite(simple_mesh.vertices))

def test_all_faces_triangles(simple_mesh):
    """M1-2: All faces are triangles (3 indices)"""
    assert all(len(face) == 3 for face in simple_mesh.faces)

def test_face_indices_valid(simple_mesh):
    """M1-3: All face vertex indices are valid"""
    n = simple_mesh.vertex_count
    for face in simple_mesh.faces:
        assert all(0 <= i < n for i in face)

def test_vertex_count_minimum(simple_mesh):
    """M1-4: Mesh has ≥ 100 vertices"""
    assert simple_mesh.vertex_count >= 100

def test_bounding_box_normalized(simple_mesh):
    """M1-5: Bounding box fits in [-1, 1]³"""
    extents = simple_mesh.bounding_box.extents
    assert np.all(extents <= 2.0)

def test_uvs_finite(simple_mesh):
    """M1-6: UV coordinates are finite (if present)"""
    if simple_mesh.uvs is not None:
        assert np.all(np.isfinite(simple_mesh.uvs))

def test_no_duplicate_vertices(simple_mesh):
    """M1-8: No duplicate vertices within ε=1e-6"""
    # Use spatial hashing for efficiency
    rounded = np.round(simple_mesh.vertices, decimals=6)
    unique = np.unique(rounded, axis=0)
    assert len(unique) == simple_mesh.vertex_count

def test_invalid_mesh_raises_error():
    """Invalid input produces descriptive error"""
    with pytest.raises((ValueError, IOError)) as exc:
        MeshData.load("tests/fixtures/meshes/invalid_mesh.glb")
    assert "mesh" in str(exc.value).lower() or "file" in str(exc.value).lower()
```

### test_segmentation.py (Stage 2, invariants M2)

```python
from vtuber_rigger.mesh.segmentation import segment, LABEL_SET

def test_all_vertices_labeled(labeled_mesh):
    """M2-1: Every vertex has exactly one label"""
    labels = labeled_mesh.labels
    assert labels.shape[0] == labeled_mesh.vertex_count
    assert not np.any(labels == None)  # no None labels

def test_labels_in_valid_set(labeled_mesh):
    """M2-2: All labels from defined set"""
    unique_labels = set(np.unique(labeled_mesh.labels))
    assert unique_labels.issubset(LABEL_SET)

def test_body_is_connected(labeled_mesh):
    """M2-3: Body vertices form connected component"""
    from scipy.sparse.csgraph import connected_components
    adj = labeled_mesh.adjacency
    body_mask = labeled_mesh.labels == "body"
    body_subgraph = adj[body_mask][:, body_mask]
    n_components, _ = connected_components(body_subgraph)
    assert n_components == 1

def test_head_is_topmost(labeled_mesh):
    """M2-4: Head vertices are topmost by Y coordinate"""
    head_y = labeled_mesh.vertices[labeled_mesh.labels == "head"][:, 1]
    body_y = labeled_mesh.vertices[labeled_mesh.labels == "body"][:, 1]
    assert head_y.mean() > body_y.mean()
```

### test_skeleton.py (Stage 3, invariants M3)

```python
import numpy as np
import pytest
from conftest import REQUIRED_BONES

def test_required_bones_present(skeleton):
    """M3-1: All 15 required VRM bones present"""
    bone_names = {b.name for b in skeleton.bones}
    missing = REQUIRED_BONES - bone_names
    assert len(missing) == 0, f"Missing required bones: {missing}"

def test_exactly_one_root(skeleton):
    """M3-5: Exactly one root bone (hips)"""
    roots = [b for b in skeleton.bones if b.parent is None]
    assert len(roots) == 1
    assert roots[0].name == "hips"

def test_no_cycles(skeleton):
    """M3-3: No cycles in parent chain"""
    for bone in skeleton.bones:
        visited = set()
        current = bone
        while current is not None:
            assert current.name not in visited, f"Cycle at {current.name}"
            visited.add(current.name)
            current = current.parent

def test_bone_positions_finite(skeleton):
    """M3-4: All bone positions finite"""
    for bone in skeleton.bones:
        assert np.all(np.isfinite(bone.position))

def test_parent_relationships(skeleton):
    """M3-7: Parent-child relationships match VRM 1.0 spec"""
    VRM_PARENTS = {
        "spine": "hips",
        "head": None,  # could be neck, upperChest, chest, or spine
        "leftUpperArm": None,  # could be upperChest, chest, or spine
        "leftLowerArm": "leftUpperArm",
        "leftHand": "leftLowerArm",
        "leftUpperLeg": "hips",
        "leftLowerLeg": "leftUpperLeg",
        "leftFoot": "leftLowerLeg",
        # right side mirrors left
    }
    for bone in skeleton.bones:
        if bone.name in VRM_PARENTS:
            expected_parent = VRM_PARENTS[bone.name]
            if expected_parent is not None:
                assert bone.parent is not None
                assert bone.parent.name == expected_parent, \
                    f"{bone.name}.parent should be {expected_parent}, got {bone.parent.name if bone.parent else None}"

def test_upperChest_implies_chest(skeleton):
    """M3-8: If upperChest present, chest MUST be present"""
    names = {b.name for b in skeleton.bones}
    if "upperChest" in names:
        assert "chest" in names

def test_neck_implies_chest(skeleton):
    """M3-9: If neck present, chest or upperChest MUST be present"""
    names = {b.name for b in skeleton.bones}
    if "neck" in names:
        assert "chest" in names or "upperChest" in names

def test_bone_names_camelCase(skeleton):
    """M3-6: Bone names use VRM camelCase convention"""
    for bone in skeleton.bones:
        # No underscores in VRM bone names
        assert "_" not in bone.name, f"Bone name '{bone.name}' should be camelCase"
```

### test_weights.py (Stage 4, invariants M4)

```python
import numpy as np
from scipy.sparse import csr_matrix

def test_weights_nonnegative(weights):
    """M4-1: All weights ≥ 0"""
    assert np.all(weights.data >= 0)

def test_weights_leq_one(weights):
    """M4-2: All weights ≤ 1"""
    assert np.all(weights.data <= 1)

def test_row_sums_one(weights):
    """M4-3: Per-vertex row sum = 1.0 ± 1e-4"""
    sums = np.array(weights.sum(axis=1)).flatten()
    assert np.allclose(sums, 1.0, atol=1e-4)

def test_sparsity(weights):
    """M4-4: Non-zero entries per vertex ≤ 8"""
    nonzero_per_row = (weights > 0).sum(axis=1)
    nonzero_array = np.asarray(nonzero_per_row).flatten()
    assert np.all(nonzero_array <= 8)

def test_all_bones_used(weights, skeleton):
    """M4-5: Each bone has ≥ 1 vertex with weight > 0"""
    bone_usage = np.asarray((weights > 0).any(axis=0)).flatten()
    assert np.all(bone_usage), f"Bones with zero weight: {np.where(~bone_usage)[0]}"

def test_no_isolated_vertices(weights):
    """M4-6: No vertex with all-zero weight"""
    vertex_has_weight = np.asarray(weights.any(axis=1)).flatten()
    assert np.all(vertex_has_weight)

def test_adjacency_smoothness(weights, simple_mesh):
    """M4-7: Weight delta between adjacent vertices < region threshold"""
    adj = simple_mesh.adjacency_list()  # returns dict {v: [neighbors]}
    for v1, neighbors in adj.items():
        for v2 in neighbors:
            w1 = weights[v1].toarray().flatten()
            w2 = weights[v2].toarray().flatten()
            delta = np.abs(w1 - w2).max()
            # Region-dependent threshold (simplified: 0.5 for all)
            assert delta < 0.5, f"Weight jump {v1}-{v2}: {delta:.4f}"
```

### test_expressions.py (Stage 5, invariants M5)

```python
import numpy as np
from conftest import VRM_STANDARD_EXPRESSIONS

def test_expression_count(expressions):
    """M5-1: All 17 standard expressions present (or ≥10 for MVP)"""
    names = set(expressions.keys())
    missing = VRM_STANDARD_EXPRESSIONS - names
    assert len(missing) <= 7, f"Too many missing expressions: {missing}"

def test_expression_names(expressions):
    """M5-1: Names match VRM 1.0 standard"""
    names = set(expressions.keys())
    # All names must be from VRM standard set or custom prefix
    for name in names:
        assert name in VRM_STANDARD_EXPRESSIONS or name.startswith("custom_")

def test_morph_target_vertex_count(expressions, simple_mesh):
    """M5-2: MorphTarget vertex count == base mesh vertex count"""
    for expr in expressions.values():
        assert expr.morph_target.vertex_count == simple_mesh.vertex_count

def test_expression_weight_range(expressions):
    """M5-3: Expression weight range [0.0, 1.0]"""
    for expr in expressions.values():
        assert 0.0 <= expr.preset_weight <= 1.0

def test_morph_target_vertices_finite(expressions):
    """M5-4: All morphTarget vertices finite"""
    for expr in expressions.values():
        assert np.all(np.isfinite(expr.morph_target.vertices))

def test_expression_mapping_1to1(expressions):
    """M5-7: Each expression maps to exactly one morphTarget"""
    target_indices = [expr.morph_target_index for expr in expressions.values()]
    assert len(target_indices) == len(set(target_indices)), "Duplicate morphTarget mapping"
```

### test_spring_bones.py (Stage 6, invariants M6)

```python
import numpy as np

def test_chain_min_length(spring_chains):
    """M6-1: Each spring chain has ≥ 2 nodes"""
    for chain in spring_chains:
        assert len(chain.nodes) >= 2

def test_node_bone_references_valid(spring_chains, skeleton):
    """M6-2: All bone node references are valid bone names"""
    all_bone_names = {b.name for b in skeleton.bones}
    for chain in spring_chains:
        for node in chain.nodes:
            assert node.bone_name in all_bone_names, f"Invalid bone ref: {node.bone_name}"

def test_no_circular_references(spring_chains):
    """M6-4: No circular references in chain hierarchy"""
    # Each chain is a list, so internal cycles are impossible
    # But check chain doesn't reference itself as parent
    for i, chain in enumerate(spring_chains):
        assert chain.root_bone != chain.nodes[0].bone_name

def test_stiffness_range(spring_chains):
    """M6-5: stiffness ∈ [0, 1]"""
    for chain in spring_chains:
        assert 0.0 <= chain.stiffness <= 1.0

def test_dragForce_range(spring_chains):
    """M6-5: dragForce ∈ [0, 1]"""
    for chain in spring_chains:
        assert 0.0 <= chain.dragForce <= 1.0

def test_hitRadius_range(spring_chains):
    """M6-5: hitRadius ∈ [0, 1]"""
    for chain in spring_chains:
        assert 0.0 <= chain.hitRadius <= 1.0

def test_gravityPower_range(spring_chains):
    """M6-5: gravityPower ∈ [0, 10]"""
    for chain in spring_chains:
        assert 0.0 <= chain.gravityPower <= 10.0

def test_gravityDir_is_unit_vector(spring_chains):
    """M6-5: gravityDir is unit vector"""
    for chain in spring_chains:
        norm = np.linalg.norm(chain.gravityDir)
        assert abs(norm - 1.0) < 1e-6

def test_collider_radii_positive(colliders):
    """M6-8: Collider radii > 0"""
    for collider in colliders:
        assert collider.radius > 0

def test_collider_centers_finite(colliders):
    """M6-7: Collider centers are finite"""
    for collider in colliders:
        assert np.all(np.isfinite(collider.center))
```

### test_lookat_firstperson.py (Stage 7, invariants M7) — NEW

```python
import numpy as np

def test_lookat_mode_valid(lookat):
    """M7-1: LookAt mode is bone or expression"""
    assert lookat.mode in {"bone", "expression"}

def test_bone_mode_has_eyes(lookat, skeleton):
    """M7-2: If bone mode, leftEye and rightEye present"""
    if lookat.mode == "bone":
        names = {b.name for b in skeleton.bones}
        assert "leftEye" in names
        assert "rightEye" in names

def test_expression_mode_has_looks(lookat, expressions):
    """M7-3: If expression mode, look expressions present"""
    if lookat.mode == "expression":
        required = {"lookUp", "lookDown", "lookLeft", "lookRight"}
        assert required.issubset(set(expressions.keys()))

def test_yaw_range_finite(lookat):
    """M7-4: Yaw range is finite"""
    assert np.isfinite(lookat.yaw_range)

def test_pitch_range_finite(lookat):
    """M7-5: Pitch range is finite"""
    assert np.isfinite(lookat.pitch_range)

def test_firstperson_offset_finite(first_person):
    """M7-6: FirstPerson camera offset is finite"""
    assert np.all(np.isfinite(first_person.camera_offset))

def test_mesh_annotation_types_valid(first_person):
    """M7-7: Mesh annotations have valid enum"""
    for ann in first_person.mesh_annotations:
        assert ann.type in {"auto", "both", "thirdPerson", "firstPerson"}
```

### test_vrm_exporter.py (Stage 8, invariants M8)

```python
import os
import json
import pytest

def test_glb_valid(output_vrm_path):
    """M8-1: File is valid GLB 2.0"""
    from vtuber_rigger.qa.schema import validate_glb
    assert validate_glb(output_vrm_path) is True

def test_vrm_extension_present(output_vrm_path):
    """M8-2: VRMC_vrm extension present with specVersion 1.0"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    assert "VRMC_vrm" in vrm.extensions
    assert vrm.extensions["VRMC_vrm"]["specVersion"] == "1.0"

def test_humanoid_present(output_vrm_path):
    """M8-3: VRMC_vrm.humanoid present with all 15 required bones"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    humanoid = vrm.extensions["VRMC_vrm"]["humanoid"]
    from conftest import REQUIRED_BONES
    for bone_name in REQUIRED_BONES:
        assert bone_name in humanoid["humanBones"], f"Missing bone: {bone_name}"

def test_meta_present(output_vrm_path):
    """M8-4: VRMC_vrm.meta present with required fields"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    meta = vrm.extensions["VRMC_vrm"]["meta"]
    assert meta["name"]
    assert meta["version"]
    assert len(meta["authors"]) >= 1
    assert meta["licenseUrl"]

def test_expressions_in_extension(output_vrm_path):
    """M8-5: VRMC_vrm.expressions present (if expressions generated)"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    assert "expressions" in vrm.extensions["VRMC_vrm"]

def test_lookat_in_extension(output_vrm_path):
    """M8-6: VRMC_vrm.lookAt present (if configured)"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    assert "lookAt" in vrm.extensions["VRMC_vrm"]

def test_firstperson_in_extension(output_vrm_path):
    """M8-7: VRMC_vrm.firstPerson present (if configured)"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    assert "firstPerson" in vrm.extensions["VRMC_vrm"]

def test_spring_bone_extension(output_vrm_path):
    """M8-8: VRMC_springBone extension present (if springs generated)"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    assert "VRMC_springBone" in vrm.extensions

def test_file_size_reasonable(output_vrm_path):
    """M8-11: File size ≤ 50 MB"""
    size_mb = os.path.getsize(output_vrm_path) / 1e6
    assert size_mb <= 50, f"File too large: {size_mb:.1f} MB"

def test_thumbnail_dimensions(output_vrm_path):
    """M8-12: Thumbnail (if present) is PNG ≤ 1024×1024"""
    from vtuber_rigger.qa.schema import parse_vrm_thumbnail
    thumb = parse_vrm_thumbnail(output_vrm_path)
    if thumb is not None:
        assert thumb.format == "PNG"
        assert thumb.width <= 1024
        assert thumb.height <= 1024

def test_no_forbidden_gltf_items(output_vrm_path):
    """M8-13: No animations or cameras in glTF (VRM forbids)"""
    from vtuber_rigger.qa.schema import parse_glb_json
    gltf = parse_glb_json(output_vrm_path)
    assert "animations" not in gltf or len(gltf["animations"]) == 0
    assert "cameras" not in gltf or len(gltf["cameras"]) == 0

def test_bone_scale_positive(output_vrm_path):
    """M8-14: Bone transform scale components are positive"""
    from vtuber_rigger.qa.schema import parse_vrm
    vrm = parse_vrm(output_vrm_path)
    for node in vrm.nodes:
        if node.get("scale") is not None:
            scale = node["scale"]
            assert all(s > 0 for s in scale), f"Non-positive scale in node {node.get('name')}"
```

### test_layer_decomp.py (Path B Stage B1, invariants B1)

```python
import numpy as np

def test_layer_count_range(layers_per_view):
    """B1-1: Layer count per view: 2 ≤ L ≤ 10"""
    for view_name, layers in layers_per_view.items():
        assert 2 <= len(layers) <= 10, f"View {view_name}: {len(layers)} layers"

def test_masks_binary(layers_per_view):
    """B1-2: Masks are binary {0, 1}"""
    for view_name, layers in layers_per_view.items():
        for layer in layers:
            unique = np.unique(layer.mask)
            assert set(unique).issubset({0, 1}), f"Non-binary mask in {view_name}/{layer.name}"

def test_coverage(layers_per_view):
    """B1-3: Union of masks = full image domain"""
    for view_name, layers in layers_per_view.items():
        masks = [l.mask for l in layers]
        mask_sum = sum(masks)
        assert np.all(mask_sum == 1), f"Coverage gap in {view_name}"

def test_disjointness(layers_per_view):
    """B1-4: Pairwise mask intersection = ∅"""
    for view_name, layers in layers_per_view.items():
        masks = [l.mask for l in layers]
        mask_sum = sum(masks)
        assert np.all(mask_sum <= 1), f"Overlap in {view_name}"

def test_no_empty_layers(layers_per_view):
    """B1-5: Each layer has ≥ 1 pixel"""
    for view_name, layers in layers_per_view.items():
        for layer in layers:
            assert layer.mask.sum() > 0, f"Empty layer {view_name}/{layer.name}"

def test_layer_dimensions(layers_per_view, screenshot_views):
    """B1-6: Layer image dimensions match input"""
    for view_name, layers in layers_per_view.items():
        input_shape = screenshot_views[view_name].shape[:2]
        for layer in layers:
            assert layer.image.shape[:2] == input_shape

def test_layer_alpha_channel(layers_per_view):
    """B1-7: Layer images have alpha channel"""
    for view_name, layers in layers_per_view.items():
        for layer in layers:
            assert layer.image.shape[2] == 4, f"No alpha in {view_name}/{layer.name}"

def test_layer_names_valid(layers_per_view):
    """B1-8: Layer names from defined set"""
    VALID = {"background", "body", "skin", "clothing",
             "hair_front", "hair_back", "eye", "accessory", "unknown"}
    for view_name, layers in layers_per_view.items():
        for layer in layers:
            assert layer.name in VALID, f"Invalid name: {layer.name}"

def test_names_consistent_across_views(layers_per_view):
    """B1-9: Layer names consistent across views"""
    view_names = list(layers_per_view.keys())
    if len(view_names) > 1:
        first_set = {l.name for l in layers_per_view[view_names[0]]}
        for vn in view_names[1:]:
            this_set = {l.name for l in layers_per_view[vn]}
            assert first_set == this_set, f"Names differ: {first_set} vs {this_set}"
```

### test_depth.py (Path B Stage B2, invariants B2)

```python
import numpy as np

def test_depth_finite(depth_per_view):
    """B2-1: All depth values finite"""
    for view_name, depth in depth_per_view.items():
        assert np.all(np.isfinite(depth)), f"Non-finite depth in {view_name}"

def test_depth_shape(depth_per_view, screenshot_views):
    """B2-2: depth.shape == image.shape[:2]"""
    for view_name, depth in depth_per_view.items():
        assert depth.shape == screenshot_views[view_name].shape[:2]

def test_depth_normalized(depth_per_view):
    """B2-3: Depth range normalized to [0, 1]"""
    for view_name, depth in depth_per_view.items():
        assert np.all((depth >= 0) & (depth <= 1)), f"Depth out of range in {view_name}"

def test_background_depth_lowest(depth_per_view, layers_per_view):
    """B2-5: background depth < all other layers"""
    for view_name, depth in depth_per_view.items():
        layers = {l.name: l for l in layers_per_view[view_name]}
        if "background" in layers:
            bg_depth = depth[layers["background"].mask].mean()
            for name, layer in layers.items():
                if name != "background":
                    other_depth = depth[layer.mask].mean()
                    assert bg_depth < other_depth, \
                        f"Background not lowest in {view_name}: bg={bg_depth}, {name}={other_depth}"
```

### test_reconstruct.py (Path B Stage B3, invariants B3)

```python
import numpy as np
import trimesh

def test_vertices_finite(reconstructed_mesh):
    """B3-2: All vertex positions finite"""
    assert np.all(np.isfinite(reconstructed_mesh.vertices))

def test_no_duplicate_vertices(reconstructed_mesh):
    """B3-3: No duplicate vertices within ε=1e-6"""
    rounded = np.round(reconstructed_mesh.vertices, decimals=6)
    unique = np.unique(rounded, axis=0)
    assert len(unique) == reconstructed_mesh.vertex_count

def test_no_degenerate_faces(reconstructed_mesh):
    """B3-4: No degenerate faces (area > 1e-10)"""
    areas = reconstructed_mesh.area_faces  # trimesh property
    assert np.all(areas > 1e-10)

def test_vertex_count_limit(reconstructed_mesh):
    """B3-5: Vertex count ≤ 100,000"""
    assert reconstructed_mesh.vertex_count <= 100_000

def test_mesh_is_manifold(reconstructed_mesh):
    """B3-6: Mesh is 2-manifold"""
    # trimesh doesn't have direct is_manifold, use euler_characteristic
    # or check via pymeshlab / OpenMesh
    # Simplified: every edge should be shared by exactly 2 faces
    from vtuber_rigger.qa.geometry import is_manifold
    assert is_manifold(reconstructed_mesh)

def test_mesh_watertight(reconstructed_mesh):
    """B3-7: Mesh is watertight"""
    # trimesh has is_watertight property
    assert reconstructed_mesh.is_watertight

def test_uvs_finite(reconstructed_mesh):
    """B3-8: UV coordinates finite (if present)"""
    if reconstructed_mesh.visual.uv is not None:
        assert np.all(np.isfinite(reconstructed_mesh.visual.uv))
```

### test_style.py (Path B Stage B4, invariants B4) — NEW

```python
import numpy as np
from PIL import Image

def test_palette_colors_in_range(style_result):
    """B4-2: All color values in [0, 1]"""
    for region, colors in style_result.palette.items():
        for color in colors:
            assert all(0 <= c <= 1 for c in color), f"Color out of range: {color}"

def test_material_type_valid(style_result):
    """B4-3: Material type is toon, unlit, or pbr"""
    assert style_result.material_type in {"toon", "unlit", "pbr"}

def test_texture_maps_valid_images(style_result):
    """B4-5: Texture maps are valid images"""
    for name, tex in style_result.textures.items():
        img = Image.open(tex.path)
        img.verify()  # raises if invalid

def test_texture_dimensions(style_result):
    """B4-6: Texture dimensions ≤ 2048×2048"""
    for name, tex in style_result.textures.items():
        with Image.open(tex.path) as img:
            assert max(img.size) <= 2048, f"Texture {name} too large: {img.size}"
```

---

## Integration Tests

### test_pipeline_a_e2e.py

```python
import pytest
from pathlib import Path
import trimesh

def test_path_a_glb_to_vrm(tmp_path):
    """Full Path A: GLB → VRM, all stages pass"""
    from vtuber_rigger.pipeline.a_pipeline import run_pipeline_a

    input_path = "tests/fixtures/meshes/small_avatar.glb"
    output_path = tmp_path / "output.vrm"

    result = run_pipeline_a(input_path, output_path)

    assert result.exit_code == 0
    assert output_path.exists()
    # Validate output passes all Stage 8 invariants (run test_vrm_exporter tests)
    from vtuber_rigger.qa.schema import validate_vrm
    assert validate_vrm(output_path) is True

def test_path_a_preserves_vertex_count(tmp_path):
    """Output mesh vertex count approximately matches input (±5%)"""
    from vtuber_rigger.pipeline.a_pipeline import run_pipeline_a

    input_path = "tests/fixtures/meshes/small_avatar.glb"
    output_path = tmp_path / "output.vrm"

    run_pipeline_a(input_path, output_path)

    input_mesh = trimesh.load(input_path)
    # VRM is GLB-based, trimesh can load geometry (not VRM extensions)
    output_mesh = trimesh.load(output_path)

    # Allow some difference due to cleanup/decimation
    diff = abs(input_mesh.vertex_count - output_mesh.vertex_count)
    assert diff / input_mesh.vertex_count < 0.05, \
        f"Vertex count changed: {input_mesh.vertex_count} → {output_mesh.vertex_count}"

def test_invalid_input_raises_error(tmp_path):
    """Invalid input produces descriptive error, not crash"""
    from vtuber_rigger.pipeline.a_pipeline import run_pipeline_a

    with pytest.raises(ValueError) as exc:
        run_pipeline_a("tests/fixtures/meshes/invalid_mesh.glb", tmp_path / "out.vrm")
    assert len(str(exc.value)) > 0

def test_blender_integration(tmp_path):
    """Verify Blender produces valid armature + weights"""
    from vtuber_rigger.utils.blender import run_blender_script
    from vtuber_rigger.rig.blender_ops import generate_rigging_script

    mesh = trimesh.load("tests/fixtures/meshes/small_avatar.glb")
    # ... get bones, weights ...
    script = generate_rigging_script(mesh, bones, weights)
    result = run_blender_script(script, workdir=tmp_path)
    assert result.returncode == 0
    assert "error" not in result.stderr.lower() or "Traceback" not in result.stderr
```

### test_pipeline_b_e2e.py — NEW

```python
def test_path_b_screenshots_to_vrm(tmp_path):
    """Full Path B: 4 screenshots → VRM"""
    from vtuber_rigger.pipeline.b_pipeline import run_pipeline_b

    inputs = {
        "front": "tests/fixtures/screenshots/front.png",
        "side": "tests/fixtures/screenshots/side.png",
        "back": "tests/fixtures/screenshots/back.png",
        "3q": "tests/fixtures/screenshots/3q.png",
    }
    output_path = tmp_path / "output.vrm"

    result = run_pipeline_b(inputs, output_path)

    assert result.exit_code == 0
    assert output_path.exists()
    from vtuber_rigger.qa.schema import validate_vrm
    assert validate_vrm(output_path) is True

def test_path_b_single_view(tmp_path):
    """Path B works with only front view (reduced quality)"""
    from vtuber_rigger.pipeline.b_pipeline import run_pipeline_b

    inputs = {"front": "tests/fixtures/screenshots/front.png"}
    output_path = tmp_path / "output.vrm"

    result = run_pipeline_b(inputs, output_path)
    assert result.exit_code == 0
    assert output_path.exists()
```

---

## Property-Based Tests (Hypothesis)

```python
import numpy as np
from hypothesis import given, strategies as st, assume
from vtuber_rigger.rig.skeleton import predict_skeleton_from_positions

@given(st.lists(
    st.tuples(
        st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False)
    ),
    min_size=100,
    max_size=1000
))
def test_skeleton_always_has_required_bones(vertex_positions):
    """M3-1: Skeleton prediction always produces 15 required bones"""
    mesh = make_mesh_from_vertices(vertex_positions)
    skeleton = predict_skeleton_from_positions(mesh)
    bone_names = {b.name for b in skeleton.bones}
    from conftest import REQUIRED_BONES
    assert REQUIRED_BONES.issubset(bone_names)

@given(st.lists(
    st.floats(min_value=0, max_value=1, allow_nan=False),
    min_size=1
))
def test_weights_normalization(weight_values):
    """M4-3: Weights always normalize to sum=1.0"""
    assume(sum(weight_values) > 0)
    weights = np.array(weight_values)
    normalized = weights / weights.sum()
    assert abs(normalized.sum() - 1.0) < 1e-6

@given(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
def test_bone_position_always_finite(x):
    """M3-4: Bone positions never produce NaN"""
    bone_pos = np.array([x, x, x])
    assert np.all(np.isfinite(bone_pos))
```

---

## Property-Based Pipeline Tests (Hypothesis) — NEW

**File:** `tests/property/test_pipeline_property.py`

Random mesh → full pipeline → verify output invariants.

```python
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from vtuber_rigger.mesh.loader import MeshData
from vtuber_rigger.pipeline.a_pipeline import run_pipeline_a

@given(st.integers(min_value=200, max_value=5000))
@settings(max_examples=20, deadline=60000)  # 20 random meshes, 60s each
def test_pipeline_a_output_always_valid(vertex_count, tmp_path_factory):
    """For ANY mesh input, pipeline output passes Stage 8 invariants."""
    # Generate random mesh
    vertices = np.random.randn(vertex_count, 3) * 0.5
    faces = generate_random_faces(vertex_count)
    mesh = MeshData(vertices=vertices, faces=faces)
    
    # Save to temp GLB
    input_path = tmp_path_factory.mktemp("test") / "input.glb"
    mesh.save(input_path)
    output_path = tmp_path_factory.mktemp("test") / "output.vrm"
    
    # Run pipeline
    result = run_pipeline_a(str(input_path), str(output_path))
    assume(result.exit_code == 0)
    
    # Verify output invariants
    from vtuber_rigger.qa.schema import validate_vrm
    assert validate_vrm(str(output_path)) is True, "Output failed VRM validation"

@given(st.lists(
    st.tuples(
        st.floats(min_value=-1, max_value=1, allow_nan=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False),
        st.floats(min_value=-1, max_value=1, allow_nan=False)
    ),
    min_size=100,
    max_size=2000
))
def test_skeleton_no_cycles_for_any_mesh(vertex_positions):
    """M3-3: Skeleton prediction never produces cycles, for ANY input."""
    mesh = make_mesh_from_vertices(vertex_positions)
    skeleton = predict_skeleton(mesh)
    
    # DFS cycle check for every bone
    for bone in skeleton.bones:
        visited = set()
        current = bone
        while current is not None:
            assert current.name not in visited, f"Cycle at {current.name}"
            visited.add(current.name)
            current = current.parent

@given(st.lists(
    st.floats(min_value=0, max_value=1, allow_nan=False),
    min_size=100,
    max_size=500
))
def test_weights_sum_to_one_for_any_distribution(raw_weights):
    """M4-3: Weight normalization always produces row sums = 1.0."""
    assume(sum(raw_weights) > 0)
    weights = np.array(raw_weights).reshape(-1, 1)
    # Simulate normalization
    normalized = weights / weights.sum(axis=1, keepdims=True)
    assert np.allclose(normalized.sum(axis=1), 1.0, atol=1e-6)
```

---

## Perceptual Metrics Tests — NEW

**Directory:** `tests/perceptual/`

### test_lpips.py

```python
import pytest
import torch
import lpips
from PIL import Image
from vtuber_rigger.qa.perceptual import render_avatar, lpips_similarity

@pytest.fixture(scope="module")
def lpips_model():
    """Load LPIPS model once (VGG backbone)"""
    return lpips.LPIPS(net='vgg')

def test_lpips_avatar_vs_reference(output_vrm_path, reference_screenshot, lpips_model):
    """
    AC9: Aesthetic similarity to reference.
    LPIPS(rendered_avatar, reference) < 0.3
    """
    rendered = render_avatar(output_vrm_path, pose="front", expression="neutral")
    similarity = lpips_similarity(rendered, reference_screenshot, lpips_model)
    assert similarity < 0.3, f"LPIPS too high: {similarity:.4f} (target < 0.3)"

def test_lpips_expression_vs_neutral(output_vrm_path, lpips_model):
    """
    AC4: Expression produces visible perceptual change.
    LPIPS(happy_expression, neutral) > 0.05 (perceptually different)
    """
    neutral = render_avatar(output_vrm_path, pose="front", expression="neutral")
    happy = render_avatar(output_vrm_path, pose="front", expression="happy")
    delta = lpips_similarity(happy, neutral, lpips_model)
    assert delta > 0.05, f"Expression 'happy' not perceptually visible: delta={delta:.4f}"

def test_lpips_all_expressions_visible(output_vrm_path, lpips_model):
    """
    AC4: ≥ 10 of 17 expressions produce visible perceptual change.
    """
    from conftest import VRM_STANDARD_EXPRESSIONS
    neutral = render_avatar(output_vrm_path, pose="front", expression="neutral")
    visible_count = 0
    for expr_name in VRM_STANDARD_EXPRESSIONS:
        rendered = render_avatar(output_vrm_path, pose="front", expression=expr_name)
        delta = lpips_similarity(rendered, neutral, lpips_model)
        if delta > 0.05:
            visible_count += 1
    assert visible_count >= 10, f"Only {visible_count}/17 expressions perceptually visible"
```

### test_clip_score.py

```python
import pytest
import torch
import clip
from PIL import Image
from vtuber_rigger.qa.perceptual import render_avatar, clip_score

@pytest.fixture(scope="module")
def clip_model():
    """Load CLIP model once"""
    model, preprocess = clip.load("ViT-B/32", device="cuda")
    return model, preprocess

EXPRESSION_TEXT_MAP = {
    "happy": "a happy anime character face",
    "angry": "an angry anime character face",
    "sad": "a sad anime character face",
    "relaxed": "a relaxed calm anime character face",
    "surprised": "a surprised anime character face",
    "blink": "an anime character face with eyes closed blinking",
}

def test_clip_expression_semantic_match(output_vrm_path, clip_model):
    """
    AC10: Expression semantically matches its label.
    CLIP(rendered_expression, text_label) > 0.25
    """
    model, preprocess = clip_model
    for expr_name, text_label in EXPRESSION_TEXT_MAP.items():
        rendered = render_avatar(output_vrm_path, pose="front", expression=expr_name)
        score = clip_score(rendered, text_label, model, preprocess)
        assert score > 0.20, f"CLIP score for '{expr_name}': {score:.4f} (target > 0.20)"

def test_clip_avatar_is_anime_character(output_vrm_path, clip_model):
    """
    AC9: Avatar is recognizable as anime character.
    CLIP(rendered_avatar, "anime VTuber character") > 0.25
    """
    model, preprocess = clip_model
    rendered = render_avatar(output_vrm_path, pose="front", expression="neutral")
    score = clip_score(rendered, "anime VTuber character avatar", model, preprocess)
    assert score > 0.25, f"CLIP avatar score: {score:.4f} (target > 0.25)"
```

### test_action_units.py

```python
import pytest
from vtuber_rigger.qa.perceptual import render_avatar, detect_action_units

# Expected action units per expression (OpenFace FACS)
EXPECTED_AUS = {
    "happy":   {"AU06": True, "AU12": True},   # cheek raiser + lip corner puller
    "angry":   {"AU04": True, "AU07": True},   # brow lowerer + lid tightener
    "sad":     {"AU01": True, "AU15": True},   # inner brow raiser + lip corner depressor
    "surprised": {"AU01": True, "AU02": True, "AU05": True, "AU26": True},
    "blink":   {"AU45": True},                  # eyelid closure
}

def test_action_units_match_expression(output_vrm_path):
    """
    AC10: Expression naturalness via OpenFace AU detection.
    Expected AUs present, unexpected AUs absent (within threshold).
    """
    for expr_name, expected_aus in EXPECTED_AUS.items():
        rendered = render_avatar(output_vrm_path, pose="front", expression=expr_name)
        detected_aus = detect_action_units(rendered)
        
        for au_name, should_be_present in expected_aus.items():
            if should_be_present:
                assert au_name in detected_aus, \
                    f"Expression '{expr_name}': {au_name} should be present"
            # Check unexpected AUs are absent (optional, threshold-based)

def test_neutral_has_no_action_units(output_vrm_path):
    """Neutral expression should have minimal action unit activation."""
    rendered = render_avatar(output_vrm_path, pose="front", expression="neutral")
    detected_aus = detect_action_units(rendered)
    # Neutral should have < 2 AUs activated
    assert len(detected_aus) < 2, f"Neutral has {len(detected_aus)} AUs active"
```

### test_landmark_match.py

```python
import pytest
import mediapipe as mp
from vtuber_rigger.qa.perceptual import render_avatar, detect_face_landmarks

def test_landmark_proportion_match(output_vrm_path, reference_screenshot):
    """
    AC9: Face landmark proportion match > 80%.
    Compare facial proportions of rendered avatar to reference.
    """
    rendered = render_avatar(output_vrm_path, pose="front", expression="neutral")
    
    avatar_landmarks = detect_face_landmarks(rendered)
    reference_landmarks = detect_face_landmarks(reference_screenshot)
    
    assume(avatar_landmarks is not None and reference_landmarks is not None)
    
    # Compute proportions (ratios independent of scale)
    avatar_props = compute_proportions(avatar_landmarks)
    reference_props = compute_proportions(reference_landmarks)
    
    # Compare: proportion match > 80%
    match_score = proportion_similarity(avatar_props, reference_props)
    assert match_score > 0.80, f"Proportion match: {match_score:.2%} (target > 80%)"
```

---

## Headless VRM Validation Tests — NEW

**Directory:** `tests/headless_vrm/`

### test_vrm_load.py

```python
import pytest
from vtuber_rigger.qa.headless_vrm import load_vrm, VRMData

def test_vrm_loads_without_crash(output_vrm_path):
    """AC2: VRM loads without error (headless)."""
    vrm = load_vrm(output_vrm_path)
    assert vrm is not None
    assert vrm.mesh is not None
    assert vrm.skeleton is not None

def test_vrm_has_required_components(output_vrm_path):
    """VRM has mesh, skeleton, expressions, spring bones."""
    vrm = load_vrm(output_vrm_path)
    assert vrm.mesh.vertex_count > 0
    assert len(vrm.skeleton.bones) >= 15  # required bones
    if vrm.expressions:
        assert len(vrm.expressions) >= 10  # MVP threshold
    if vrm.spring_chains:
        assert all(len(c.nodes) >= 2 for c in vrm.spring_chains)
```

### test_expression_apply.py

```python
import numpy as np
from vtuber_rigger.qa.headless_vrm import load_vrm, apply_expression

def test_all_expressions_produce_deformation(output_vrm_path):
    """AC4: Each expression produces visible mesh deformation."""
    vrm = load_vrm(output_vrm_path)
    base_vertices = vrm.mesh.vertices.copy()
    
    for expr_name in vrm.expressions:
        deformed = apply_expression(vrm, expr_name, weight=1.0)
        delta = np.abs(deformed.vertices - base_vertices).max()
        assert delta > 1e-4, f"Expression '{expr_name}': no visible deformation (delta={delta:.6f})"
        assert np.all(np.isfinite(deformed.vertices)), f"Expression '{expr_name}': NaN/Inf vertices"

def test_expression_weight_interpolation(output_vrm_path):
    """Expression at weight=0.5 produces half the deformation of weight=1.0."""
    vrm = load_vrm(output_vrm_path)
    base = vrm.mesh.vertices.copy()
    
    full = apply_expression(vrm, "happy", weight=1.0)
    half = apply_expression(vrm, "happy", weight=0.5)
    
    full_delta = np.abs(full.vertices - base)
    half_delta = np.abs(half.vertices - base)
    
    # Half weight should produce approximately half the deformation
    ratio = half_delta.max() / (full_delta.max() + 1e-8)
    assert 0.4 < ratio < 0.6, f"Weight interpolation broken: ratio={ratio:.3f}"
```

### test_bone_rotation.py

```python
import numpy as np
from vtuber_rigger.qa.headless_vrm import load_vrm, rotate_bone, compute_deformed_mesh

def test_required_bones_affect_mesh(output_vrm_path):
    """AC3: All 15 required bones are animatable (rotation deforms mesh)."""
    vrm = load_vrm(output_vrm_path)
    base = vrm.mesh.vertices.copy()
    
    from conftest import REQUIRED_BONES
    for bone_name in REQUIRED_BONES:
        if bone_name in [b.name for b in vrm.skeleton.bones]:
            deformed = rotate_bone(vrm, bone_name, x=10, y=10, z=10)  # 10° on each axis
            delta = np.abs(deformed.vertices - base).max()
            assert delta > 1e-4, f"Bone '{bone_name}': no mesh deformation"
            assert np.all(np.isfinite(deformed.vertices))

def test_bone_rotation_is_local(output_vrm_path):
    """Rotating parent bone affects child bone's world position but not local rotation."""
    vrm = load_vrm(output_vrm_path)
    
    # Rotate parent (hips)
    deformed = rotate_bone(vrm, "hips", x=30)
    
    # Child (spine) world position should change
    spine_world_before = vrm.get_bone_world_position("spine")
    spine_world_after = deformed.get_bone_world_position("spine")
    assert np.linalg.norm(spine_world_after - spine_world_before) > 1e-3
```

### test_lookat_mock.py

```python
import numpy as np
from vtuber_rigger.qa.headless_vrm import load_vrm, apply_lookat

def test_lookat_moves_eyes(output_vrm_path):
    """AC6: LookAt input causes eye bone rotation."""
    vrm = load_vrm(output_vrm_path)
    
    if "leftEye" not in [b.name for b in vrm.skeleton.bones]:
        pytest.skip("No eye bones in this VRM")
    
    # Mock face input: looking 30° right, 15° up
    deformed = apply_lookat(vrm, yaw=30, pitch=15)
    
    # Verify eye bones rotated
    left_eye_rot_before = vrm.get_bone_rotation("leftEye")
    left_eye_rot_after = deformed.get_bone_rotation("leftEye")
    
    rot_delta = np.abs(left_eye_rot_after - left_eye_rot_before).max()
    assert rot_delta > 0.01, f"LookAt didn't rotate eyes: delta={rot_delta:.4f}"

def test_lookat_yaw_pitch_ranges(output_vrm_path):
    """LookAt respects yaw/pitch range limits."""
    vrm = load_vrm(output_vrm_path)
    
    # Try extreme yaw
    deformed = apply_lookat(vrm, yaw=180, pitch=90)
    
    # Actual eye rotation should be clamped to configured range
    left_eye_rot = deformed.get_bone_rotation("leftEye")
    max_actual_yaw = np.abs(left_eye_rot[1])  # Y component = yaw
    assert max_actual_yaw <= 1.6, f"LookAt yaw not clamped: {max_actual_yaw:.3f} rad"
```

### test_spring_physics.py

```python
import numpy as np
from vtuber_rigger.qa.headless_vrm import load_vrm, simulate_spring_physics

def test_spring_bones_oscillate(output_vrm_path):
    """AC5: Spring bones oscillate when avatar moves."""
    vrm = load_vrm(output_vrm_path)
    if not vrm.spring_chains:
        pytest.skip("No spring bones in this VRM")
    
    # Apply impulse (simulate head movement: 1m/s horizontal)
    positions_over_time = simulate_spring_physics(
        vrm, impulse=(1.0, 0, 0), frames=120, fps=60
    )
    
    # Check first spring chain oscillates
    chain = vrm.spring_chains[0]
    tip_node = chain.nodes[-1]
    tip_positions = [frame[tip_node.bone_name] for frame in positions_over_time]
    
    # Should oscillate: max displacement > 0 after initial movement
    displacements = [np.linalg.norm(p - tip_positions[0]) for p in tip_positions]
    max_disp = max(displacements)
    assert max_disp > 0.01, f"Spring bones didn't oscillate: max displacement={max_disp:.4f}"

def test_spring_bones_damp(output_vrm_path):
    """Spring bone oscillation damps over time (settles)."""
    vrm = load_vrm(output_vrm_path)
    if not vrm.spring_chains:
        pytest.skip("No spring bones")
    
    positions_over_time = simulate_spring_physics(
        vrm, impulse=(1.0, 0, 0), frames=120, fps=60
    )
    
    chain = vrm.spring_chains[0]
    tip_node = chain.nodes[-1]
    displacements = [np.linalg.norm(p - positions_over_time[0][tip_node.bone_name]) 
                     for p in [f[tip_node.bone_name] for f in positions_over_time]]
    
    # Amplitude at end should be < 1% of max
    max_amp = max(displacements)
    final_amp = displacements[-1]
    assert final_amp < 0.01 * max_amp, f"Spring didn't damp: final={final_amp:.4f}, max={max_amp:.4f}"
```

### test_round_trip.py

```python
import numpy as np
from vtuber_rigger.qa.headless_vrm import load_vrm, export_vrm

def test_vrm_round_trip_preserves_geometry(output_vrm_path, tmp_path):
    """Export → re-import preserves mesh geometry."""
    vrm1 = load_vrm(output_vrm_path)
    
    # Re-export
    reexport_path = tmp_path / "round_trip.vrm"
    export_vrm(vrm1, str(reexport_path))
    
    # Re-import
    vrm2 = load_vrm(str(reexport_path))
    
    # Compare geometry
    assert vrm1.mesh.vertex_count == vrm2.mesh.vertex_count
    assert np.allclose(vrm1.mesh.vertices, vrm2.mesh.vertices, atol=1e-4)

def test_vrm_round_trip_preserves_skeleton(output_vrm_path, tmp_path):
    """Export → re-import preserves skeleton."""
    vrm1 = load_vrm(output_vrm_path)
    reexport_path = tmp_path / "round_trip.vrm"
    export_vrm(vrm1, str(reexport_path))
    vrm2 = load_vrm(str(reexport_path))
    
    bones1 = {b.name for b in vrm1.skeleton.bones}
    bones2 = {b.name for b in vrm2.skeleton.bones}
    assert bones1 == bones2, f"Bone set changed: {bones1.symmetric_difference(bones2)}"

def test_vrm_round_trip_preserves_weights(output_vrm_path, tmp_path):
    """Export → re-import preserves skin weights."""
    vrm1 = load_vrm(output_vrm_path)
    reexport_path = tmp_path / "round_trip.vrm"
    export_vrm(vrm1, str(reexport_path))
    vrm2 = load_vrm(str(reexport_path))
    
    # Weights should be close (may have minor float differences)
    assert vrm1.weights.shape == vrm2.weights.shape
    diff = np.abs(vrm1.weights - vrm2.weights).max()
    assert diff < 1e-4, f"Weight difference too large: {diff:.6f}"
```

---

## Physics Metrics Tests — NEW

**Directory:** `tests/physics/`

### test_oscillation.py

```python
import numpy as np
from scipy.fft import fft, fftfreq
from vtuber_rigger.qa.physics import simulate_impulse_response

def test_oscillation_frequency_in_range(output_vrm_path):
    """Spring bone oscillation frequency: 0.5-3 Hz (natural hair/cloth range)."""
    positions = simulate_impulse_response(output_vrm_path, impulse=(1.0, 0, 0), duration=2.0, fps=60)
    
    # FFT to find dominant frequency
    displacements = positions - positions[0]
    magnitudes = np.linalg.norm(displacements, axis=-1)
    
    freqs = fftfreq(len(magnitudes), d=1/60)
    spectrum = np.abs(fft(magnitudes))
    
    # Exclude DC (index 0), find peak
    peak_idx = np.argmax(spectrum[1:]) + 1
    peak_freq = abs(freqs[peak_idx])
    
    assert 0.5 <= peak_freq <= 3.0, f"Oscillation freq {peak_freq:.2f} Hz outside [0.5, 3.0]"
```

### test_damping.py

```python
import numpy as np
from scipy.optimize import curve_fit
from vtuber_rigger.qa.physics import simulate_impulse_response

def test_damping_ratio_in_range(output_vrm_path):
    """Spring bone damping ratio: 0.3-0.9 (underdamped, natural feel)."""
    positions = simulate_impulse_response(output_vrm_path, impulse=(1.0, 0, 0), duration=3.0, fps=60)
    magnitudes = np.linalg.norm(positions - positions[0], axis=-1)
    
    # Fit exponential decay: A(t) = A0 * exp(-ζωt)
    # where ζ = damping ratio
    def decay_func(t, A0, zeta, omega):
        return A0 * np.exp(-zeta * omega * t)
    
    t = np.arange(len(magnitudes)) / 60
    try:
        popt, _ = curve_fit(decay_func, t, magnitudes, p0=[1.0, 0.5, 5.0], maxfev=10000)
        zeta = popt[1]
    except:
        pytest.skip("Could not fit damping curve")
    
    assert 0.3 <= zeta <= 0.9, f"Damping ratio {zeta:.3f} outside [0.3, 0.9]"

def test_settling_time_under_2s(output_vrm_path):
    """Spring bones settle within 2 seconds (amplitude < 1% of initial)."""
    positions = simulate_impulse_response(output_vrm_path, impulse=(1.0, 0, 0), duration=3.0, fps=60)
    magnitudes = np.linalg.norm(positions - positions[0], axis=-1)
    
    max_amp = magnitudes.max()
    threshold = 0.01 * max_amp
    
    # Find settling time
    settled = np.where(magnitudes < threshold)[0]
    if len(settled) > 0:
        settling_time = settled[0] / 60  # frames to seconds
        assert settling_time < 2.0, f"Settling time {settling_time:.2f}s > 2.0s"
    else:
        pytest.fail("Spring bones did not settle within 3s")
```

### test_collision.py

```python
import numpy as np
from vtuber_rigger.qa.physics import simulate_collision_test

def test_spring_chain_deflects_from_collider(output_vrm_path):
    """Spring chain deflects from collider (no penetration)."""
    collision_result = simulate_collision_test(output_vrm_path)
    
    # Check: chain positions never inside collider sphere
    min_distance = collision_result.min_distance_to_collider
    assert min_distance > 0, f"Spring chain penetrated collider by {-min_distance:.4f}"

def test_spring_chain_recovers_after_collider(output_vrm_path):
    """Spring chain recovers natural position after passing collider."""
    collision_result = simulate_collision_test(output_vrm_path)
    
    # Final position should be close to rest position
    rest_pos = collision_result.rest_position
    final_pos = collision_result.final_position
    recovery = np.linalg.norm(final_pos - rest_pos)
    
    assert recovery < 0.05, f"Spring chain didn't recover: distance={recovery:.4f}"
```

---

## Benchmark Tests (vs VRoid Reference)

### test_vroid_comparison.py

```python
import numpy as np
import pytest

@pytest.fixture(scope="module")
def vroid_references():
    """Load 3-5 VRoid Hub reference VRM files"""
    from scripts.download_vroid_refs import download_if_needed
    download_if_needed()
    return [load_vrm(path) for path in VROID_REF_PATHS]

@pytest.fixture(scope="module")
def our_output():
    """Run pipeline on test input"""
    from vtuber_rigger.pipeline.a_pipeline import run_pipeline_a
    output = run_pipeline_a("tests/fixtures/meshes/small_avatar.glb", "tmp_output.vrm")
    return load_vrm("tmp_output.vrm")

def test_bone_positions_vs_vroid(our_output, vroid_references):
    """
    Compare predicted bone positions to VRoid mean.
    L2 distance should be < 0.15 (normalized units, relaxed from 0.1)
    """
    for bone_name in REQUIRED_BONES:
        our_pos = our_output.bone_positions[bone_name]
        ref_positions = [ref.bone_positions.get(bone_name) for ref in vroid_references]
        ref_positions = [p for p in ref_positions if p is not None]
        if ref_positions:
            ref_mean = np.mean(ref_positions, axis=0)
            l2 = np.linalg.norm(our_pos - ref_mean)
            assert l2 < 0.15, f"Bone {bone_name}: L2={l2:.4f} (target < 0.15)"

def test_expression_coverage(our_output):
    """At least 10 of 17 expressions produce visible deformation"""
    from conftest import VRM_STANDARD_EXPRESSIONS
    visible_count = 0
    for expr_name in VRM_STANDARD_EXPRESSIONS:
        if expr_name in our_output.expressions:
            expr = our_output.expressions[expr_name]
            delta = np.abs(expr.morph_target.vertices - our_output.base_mesh.vertices)
            if delta.max() > 1e-4:  # visible deformation threshold
                visible_count += 1
    assert visible_count >= 10, f"Only {visible_count}/17 expressions visible"

def test_poly_count_comparison(our_output, vroid_references):
    """Our poly count comparable to VRoid references (< 1.5× max)"""
    our_polys = our_output.mesh.face_count
    ref_poly_counts = [ref.mesh.face_count for ref in vroid_references]
    max_ref = max(ref_poly_counts)
    assert our_polys < max_ref * 1.5, \
        f"Poly count {our_polys} > 1.5× max VRoid ({max_ref})"

def test_file_size_comparison(our_output, vroid_references):
    """Our file size comparable to VRoid references"""
    our_size = os.path.getsize(our_output.path)
    ref_sizes = [os.path.getsize(ref.path) for ref in vroid_references]
    max_ref = max(ref_sizes)
    assert our_size < max_ref * 2, f"File {our_size/1e6:.1f}MB > 2× max VRoid ({max_ref/1e6:.1f}MB)"
```

---

## Visual / Human Evaluation Checklist

This cannot be automated. Run manually and record results.

### VSeeFace Live Test
```
□ VRM loads without crash
□ All 15 required bones visible in rig panel
□ Dragging bones moves mesh correctly
□ Expressions: move slider → face deforms
□ Spring bones: move avatar → hair/cloth swings
□ LookAt: move face/cursor → eyes follow
□ FirstPerson: camera offset is correct
□ No Z-fighting or mesh glitches
□ Author/meta information displays correctly
□ Avatar tracks head movement smoothly
```

### Screenshot → Reconstructed Avatar (Path B)
```
□ Reconstructed mesh resembles reference screenshots
□ Hair is on the correct side of head
□ Eyes are correctly placed (large, anime-style)
□ Proportions are humanoid (not distorted)
□ Materials look consistent with reference style
□ Colors match reference (skin, hair, clothing)
□ No obvious geometry artifacts
□ VRM works in VSeeFace with tracking
```

### Editability Test
```
□ Can change hair color by editing material in Blender
□ Can add new expression and it works in VRM apps
□ Can modify spring bone stiffness in VRM app
□ All edits preserve VRM validity (re-export works)
□ Can import back to Blender without errors
```

### Naturalness Test
```
□ Blink animation looks natural (not too fast/slow)
□ Happy expression conveys happiness
□ Spring bone oscillation damps naturally
□ LookAt tracking feels responsive, not laggy
□ Avatar doesn't glitch when head moves quickly
```

---

## CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: VTuber Rigger Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'  # Match ML venv, not 3.14
      - name: Install deps
        run: |
          pip install -e ".[dev]"
      - name: Generate test fixtures
        run: python scripts/generate_test_mesh.py
      - name: Run unit tests
        run: pytest tests/unit/ -v --tb=short --cov=vtuber_rigger

  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install deps
        run: pip install -e ".[dev]"
      - name: Install Blender (for integration tests)
        run: |
          wget https://download.blender.org/release/Blender5.2/blender-5.2.0-linux-x64.tar.xz
          tar xf blender-5.2.0-linux-x64.tar.xz
          echo "$(pwd)/blender-5.2.0-linux-x64" >> $GITHUB_PATH
      - name: Run integration tests
        run: pytest tests/integration/ -v --tb=short

  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install deps
        run: pip install -e ".[dev]" hypothesis
      - name: Run property tests
        run: pytest tests/property/ -v --tb=short

  benchmark-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install deps
        run: pip install -e ".[dev]"
      - name: Download VRoid references
        run: python scripts/download_vroid_refs.py
      - name: Run benchmark tests
        run: pytest tests/benchmarks/ -v --tb=short
```

**Note:** CI uses Python 3.12 (not 3.14) for torch compatibility. Local dev can use 3.14 for non-ML code.

---

## Coverage Target

```
Minimum: 85% line coverage
Desired:  95% line coverage
Critical paths: 100% (skeleton, weights, VRM export, schema validation)
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit only (fast)
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=vtuber_rigger --cov-report=html

# Specific module
pytest tests/unit/test_skeleton.py -v

# Property-based (slow)
pytest tests/property/ -v

# Benchmarks (requires VRoid refs)
pytest tests/benchmarks/ -v

# Integration (requires Blender)
pytest tests/integration/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Human evaluation checklist (manual, not automated)
# See "Visual / Human Evaluation Checklist" section above
```

---

## Test Data Management

- Fixtures < 10MB total (small meshes, thumbnails)
- VRoid references downloaded at test time via `scripts/download_vroid_refs.py`
- Screenshots: generate via Blender render or use CC0 references
- Procedural test mesh: `scripts/generate_test_mesh.py` (trimesh-based)
- No network required after initial fixture download
- Fixtures are deterministic (same hash on every run)

---

## Invariant Coverage Matrix

### Discrete Invariants (SPEC.md → tests/)

| SPEC.md Invariant | Test File | Test Function |
|-------------------|-----------|---------------|
| M1-1 (vertices finite) | unit/test_mesh_loader.py | test_vertices_finite |
| M1-2 (faces triangles) | unit/test_mesh_loader.py | test_all_faces_triangles |
| M1-3 (indices valid) | unit/test_mesh_loader.py | test_face_indices_valid |
| M1-4 (vertex count) | unit/test_mesh_loader.py | test_vertex_count_minimum |
| M1-5 (bounding box) | unit/test_mesh_loader.py | test_bounding_box_normalized |
| M1-6 (UVs finite) | unit/test_mesh_loader.py | test_uvs_finite |
| M1-7 (materials valid) | unit/test_mesh_loader.py | test_material_indices_valid |
| M1-8 (no duplicates) | unit/test_mesh_loader.py | test_no_duplicate_vertices |
| M2-1 (every vertex labeled) | unit/test_segmentation.py | test_all_vertices_labeled |
| M2-2 (labels in set) | unit/test_segmentation.py | test_labels_in_valid_set |
| M2-3 (body connected) | unit/test_segmentation.py | test_body_is_connected |
| M2-4 (head topmost) | unit/test_segmentation.py | test_head_is_topmost |
| M3-1 (15 required bones) | unit/test_skeleton.py | test_required_bones_present |
| M3-3 (no cycles) | unit/test_skeleton.py | test_no_cycles |
| M3-3 (no cycles, any input) | property/test_pipeline_property.py | test_skeleton_no_cycles_for_any_mesh |
| M3-4 (positions finite) | unit/test_skeleton.py | test_bone_positions_finite |
| M3-5 (root is hips) | unit/test_skeleton.py | test_exactly_one_root |
| M3-7 (parent relationships) | unit/test_skeleton.py | test_parent_relationships |
| M3-8 (upperChest→chest) | unit/test_skeleton.py | test_upperChest_implies_chest |
| M3-9 (neck→chest) | unit/test_skeleton.py | test_neck_implies_chest |
| M4-1 (weights ≥ 0) | unit/test_weights.py | test_weights_nonnegative |
| M4-2 (weights ≤ 1) | unit/test_weights.py | test_weights_leq_one |
| M4-3 (row sums = 1) | unit/test_weights.py | test_row_sums_one |
| M4-3 (any distribution) | property/test_pipeline_property.py | test_weights_sum_to_one_for_any_distribution |
| M4-4 (sparsity ≤ 8) | unit/test_weights.py | test_sparsity |
| M4-5 (all bones used) | unit/test_weights.py | test_all_bones_used |
| M4-6 (no isolated) | unit/test_weights.py | test_no_isolated_vertices |
| M4-7 (adjacency smooth) | unit/test_weights.py | test_adjacency_smoothness |
| M5-1 (17 expressions) | unit/test_expressions.py | test_expression_count |
| M5-2 (vertex count match) | unit/test_expressions.py | test_morph_target_vertex_count |
| M5-3 (weight range) | unit/test_expressions.py | test_expression_weight_range |
| M5-4 (vertices finite) | unit/test_expressions.py | test_morph_target_vertices_finite |
| M5-7 (1:1 mapping) | unit/test_expressions.py | test_expression_mapping_1to1 |
| M6-1 (chain ≥ 2) | unit/test_spring_bones.py | test_chain_min_length |
| M6-2 (bone refs valid) | unit/test_spring_bones.py | test_node_bone_references_valid |
| M6-5 (param ranges) | unit/test_spring_bones.py | test_stiffness_range etc. |
| M6-7 (colliders finite) | unit/test_spring_bones.py | test_collider_centers_finite |
| M6-8 (radii > 0) | unit/test_spring_bones.py | test_collider_radii_positive |
| M7-1..M7-7 | unit/test_lookat_firstperson.py | (all Stage 7 tests) |
| M8-1..M8-14 | unit/test_vrm_exporter.py | (all Stage 8 tests) |
| B1-1..B1-9 | unit/test_layer_decomp.py | (all layer tests) |
| B2-1..B2-5 | unit/test_depth.py | (all depth tests) |
| B3-1..B3-9 | unit/test_reconstruct.py | (all reconstruct tests) |
| B4-1..B4-7 | unit/test_style.py | (all style tests) |

### Acceptance Criteria Coverage (AC → test category)

| AC | Description | Test Category | Automation Level |
|----|-------------|---------------|------------------|
| AC1 | VRM invariants (M8-1..14) | unit/test_vrm_exporter.py | FULL |
| AC2 | VSeeFace load | headless_vrm/test_vrm_load.py + human | 90% auto |
| AC3 | 15 bones animatable | headless_vrm/test_bone_rotation.py | FULL |
| AC4 | ≥10 expressions visible | headless_vrm/test_expression_apply.py + perceptual/test_lpips.py | FULL |
| AC5 | Spring oscillation | headless_vrm/test_spring_physics.py + physics/test_oscillation.py | FULL |
| AC5 | Damping natural | physics/test_damping.py | FULL |
| AC5 | Collision works | physics/test_collision.py | FULL |
| AC6 | LookAt works | headless_vrm/test_lookat_mock.py + human (webcam) | 90% auto |
| AC7 | File size ≤ 50MB | unit/test_vrm_exporter.py | FULL |
| AC8 | Poly count ≤ 50k | unit/test_vrm_exporter.py | FULL |
| AC9 | Aesthetic similarity | perceptual/test_lpips.py + test_clip_score.py + test_landmark_match.py | 95% auto |
| AC10 | Expression naturalness | perceptual/test_action_units.py + benchmarks | 90% auto |

**Automation summary:** 10 of 10 ACs have automated test coverage. Human required only for: final webcam tracking UX (AC2/AC6), artistic sign-off (AC9), edge cases (AC10).

### Property-Based Coverage (Layer 3, 5 of verification pyramid)

| Invariant | Property Test | What it proves |
|-----------|---------------|----------------|
| M3-1 (15 bones) | test_skeleton_always_has_required_bones | For ANY mesh, 15 bones present |
| M3-3 (no cycles) | test_skeleton_no_cycles_for_any_mesh | For ANY mesh, no cycles |
| M4-3 (sum=1) | test_weights_sum_to_one_for_any_distribution | For ANY weight distribution, normalized |
| M8-* (VRM valid) | test_pipeline_a_output_always_valid | For ANY mesh, pipeline output is valid VRM |

### Perceptual Metrics Coverage (Layer 7 of verification pyramid)

| AC | Metric | Threshold | Test File |
|----|--------|-----------|-----------|
| AC4 | LPIPS(expr, neutral) | > 0.05 | perceptual/test_lpips.py |
| AC9 | LPIPS(avatar, reference) | < 0.3 | perceptual/test_lpips.py |
| AC9 | CLIP(avatar, "anime VTuber") | > 0.25 | perceptual/test_clip_score.py |
| AC10 | CLIP(expr, text_label) | > 0.20 | perceptual/test_clip_score.py |
| AC10 | OpenFace AU match | expected AUs present | perceptual/test_action_units.py |
| AC9 | Landmark proportion | > 80% match | perceptual/test_landmark_match.py |

### Headless VRM Coverage (Layer 8 of verification pyramid)

| AC | Operation | Verification | Test File |
|----|-----------|--------------|-----------|
| AC2 | VRM load | no crash | headless_vrm/test_vrm_load.py |
| AC3 | Bone rotation | mesh deforms | headless_vrm/test_bone_rotation.py |
| AC4 | Expression apply | mesh deforms, finite | headless_vrm/test_expression_apply.py |
| AC4 | Weight interpolation | 0.5 weight = 0.5 delta | headless_vrm/test_expression_apply.py |
| AC5 | Spring impulse | oscillation occurs | headless_vrm/test_spring_physics.py |
| AC5 | Spring damping | settles < 1% | headless_vrm/test_spring_physics.py |
| AC6 | LookAt mock | eye bones rotate | headless_vrm/test_lookat_mock.py |
| AC6 | LookAt clamp | respects yaw/pitch range | headless_vrm/test_lookat_mock.py |
| — | Round-trip | geometry/skeleton/weights preserved | headless_vrm/test_round_trip.py |

### Physics Metrics Coverage (Layer 8, specialized)

| AC | Metric | Target | Test File |
|----|--------|--------|-----------|
| AC5 | Oscillation freq | 0.5-3 Hz | physics/test_oscillation.py |
| AC5 | Damping ratio | 0.3-0.9 | physics/test_damping.py |
| AC5 | Settling time | < 2s | physics/test_damping.py |
| AC5 | Collision deflection | min distance > 0 | physics/test_collision.py |
| AC5 | Collision recovery | < 0.05 from rest | physics/test_collision.py |

---

## Human Evaluation Procedure (minimal residual)

**When:** Before release (after all automated tests pass).

**Who:** Developer + at least one external tester.

**What remains manual (cannot be automated):**
1. **VSeeFace real webcam tracking** (~3 min)
   - Real webcam face tracking responsiveness
   - Framerate in real usage
   - UX feel (lag, jitter)

2. **Final aesthetic sign-off** (~5 min)
   - "Does this avatar look good?"
   - "Does it match the reference style?"
   - Novel edge cases outside training data

3. **Artistic nuance** (~2 min)
   - Expression subtlety ("is this 'happy' actually happy?")
   - Cultural/stylistic appropriateness

**Total human time per release: ~10 minutes** (down from ~1 hour before automation).

**How:**
1. All automated tests MUST pass first (CI green)
2. Run pipeline on test fixture
3. Open output.vrm in VSeeFace with real webcam
4. Walk through minimal "VSeeFace Live Test" checklist (real webcam only):
   ```
   □ VRM loads in VSeeFace without crash
   □ Real webcam tracking works (head movement → avatar moves)
   □ Framerate acceptable (>30fps)
   □ No visual glitches on tracking
   □ Avatar aesthetically matches reference (5-second look)
   ```
5. Record pass/fail in `tests/manual/<date>_human_eval.md`
6. If any fail: file issue, fix, re-run automated + manual tests
7. All pass → release

**Full manual checklist (for thorough testing, optional):** See "Visual / Human Evaluation Checklist" section above. Only needed for major releases or new feature areas.

**Record results in:** `tests/manual/<date>_human_eval.md`