# VTuber Rigger — SPEC.md
**Formal specification with discrete invariants**
**VRM 1.0 compliant** | Revision 2

---

## Scope

VTuber Rigger converts either:
- **(Path A)** A 3D model file (GLB/VRM/FBX, no rig) → VRM 1.0 with full rig
- **(Path B)** Reference screenshots of an existing VTuber → VRM 1.0 with full rig

Output: VRM 1.0 file (`.vrm` — glTF 2.0 binary with `VRMC_vrm` extension)

Target applications: VTube Studio, VSeeFace, VRMViewer, any VRM 1.0-compatible app.

---

## Glossary

| Term | Definition |
|------|------------|
| VRM 1.0 | Virtual Reality Model — glTF 2.0 extension for avatars (by VRM Consortium) |
| `VRMC_vrm` | Main VRM 1.0 extension (contains humanoid, meta, expressions, lookAt, firstPerson) |
| `VRMC_springBone` | Separate extension for hair/cloth physics chains |
| `VRMC_node_constraint` | Separate extension for twist/aim/rotation constraints |
| `VRMC_materials_mtoon` | Toon shader material extension |
| T-pose | Default VRM pose: arms extended horizontally, legs straight |
| Expression | VRM 1.0 term for blend shape / morph target (facial expression) |
| Spring bone | Physics chain for hair/cloth dynamics |
| Skin weight | Per-vertex bone influence (0–1, sum to 1.0) |
| LookAt | Eye gaze control (yaw/pitch) |
| Humanoid bones | Standard VRM 1.0 humanoid bone set (15 required + optional) |

---

## VRM 1.0 Reference

**Specification source:** https://github.com/vrm-c/vrm-specification
**Extension name:** `VRMC_vrm` (specVersion "1.0")
**File format:** `.vrm` (GLB 2.0 binary with VRM extensions)
**Coordinate system:** glTF 2.0 (right-handed, Y-up, Z-forward)

**Required VRMC_vrm subfields:**
- `humanoid` — bone assignment (required)
- `meta` — model metadata (required)

**Optional VRMC_vrm subfields:**
- `firstPerson` — first-person camera settings
- `lookAt` — eye gaze control
- `expressions` — facial expressions (blend shapes)

**Companion extensions:**
- `VRMC_springBone` — hair/cloth physics
- `VRMC_node_constraint` — twist/aim/rotation constraints
- `VRMC_materials_mtoon` — toon shader material
- `KHR_materials_unlit` — unlit material
- `KHR_texture_transform` — texture UV transform

---

## VRM 1.0 Humanoid Bones (authoritative)

### Required Bones (15)

```
Torso:
  hips        (root, no parent)
  spine       (parent: hips)
  head         (parent: neck if exists, else chest if exists, else spine)

Arms (×2, left/right):
  {side}UpperArm  (parent: chest or upperChest or spine)
  {side}LowerArm  (parent: {side}UpperArm)
  {side}Hand      (parent: {side}LowerArm)

Legs (×2, left/right):
  {side}UpperLeg  (parent: hips)
  {side}LowerLeg  (parent: {side}UpperLeg)
  {side}Foot      (parent: {side}LowerLeg)

side ∈ {left, right}
```

### Optional Bones

```
Torso (optional, in hierarchy order):
  chest        (parent: spine)
  upperChest   (parent: chest, requires chest)
  neck         (parent: upperChest or chest, requires upperChest/chest)

Head (optional):
  leftEye      (parent: head)
  rightEye     (parent: head)
  jaw          (parent: head)

Legs (optional):
  leftToes    (parent: leftFoot)
  rightToes   (parent: rightFoot)

Fingers (optional, all 5 × 2 sides):
  {side}Thumb  {metacarpal, proximal, distal}  (3 phalanges)
  {side}Index  {metacarpal, proximal, distal}  (3 phalanges)
  {side}Middle {metacarpal, proximal, distal}  (3 phalanges)
  {side}Ring   {metacarpal, proximal, distal}  (3 phalanges)
  {side}Little {metacarpal, proximal, distal}  (3 phalanges)
```

### Parent-Child Relationships (invariant)

```
hips → spine → [chest → [upperChest → [neck →]]] head → [eyes, jaw]
                ↘ {side}UpperArm → {side}LowerArm → {side}Hand → [fingers]
hips → {side}UpperLeg → {side}LowerLeg → {side}Foot → [toes]
```

**Rules:**
- `upperChest` requires `chest` to exist
- `neck` requires `upperChest` or `chest` to exist
- Each bone has exactly one parent (except `hips` = root)
- No cycles (acyclic graph)

---

## VRM 1.0 Expressions (replaces "blend shapes")

### Predefined Expressions (17 standard)

```
Emotion (5):
  happy, angry, sad, relaxed, surprised

Look (5):
  lookUp, lookDown, lookLeft, lookRight, blink

Blink split (2):
  blinkLeft, blinkRight

Vowel mouth (5):
  aa, ih, ou, ee, oh
```

**Neutral** is implicit (all expression weights = 0 → neutral face).

### Expression Properties

- Each expression has `weight ∈ [0.0, 1.0]`
- Each expression targets morphTarget on mesh (via `meshes[*].extras.targetNames`)
- Multiple expressions can be active simultaneously
- Total vertex displacement = sum of (expression_weight × morphTarget_delta)

---

## System Invariants (always true, no exceptions)

### S1: Pipeline Termination
```
Every stage completes in finite time.
No infinite loops. No unbounded recursion.
Timeout per stage: 60s (configurable).
```

### S2: Data Integrity
```
All data passed between stages is serializable and deserializable
without loss of structure.
Intermediate representations are versioned.
```

### S3: VRM Compliance
```
Every output file:
  - Parses as valid GLB 2.0 (passes glTF-Validator)
  - Contains VRMC_vrm extension with specVersion "1.0"
  - Has all required VRMC_vrm subfields (humanoid, meta)
  - Passes VRM 1.0 JSON schema validation
  - Opens in VSeeFace / VTube Studio without errors
```

### S4: No External Dependencies at Runtime
```
All ML models, weights, and data are local.
No network calls during processing.
Reference data pre-downloaded at install time.
```

---

## Stage Specifications

---

### Stage 1: Mesh Import (Path A)

**Input:** GLB / GLTF / VRM / FBX file

**Output:** Normalized triangle mesh with per-vertex normals, UV coordinates, material indices

**Discrete Invariants (M1):**

```
M1-1: All vertex positions are 3D finite points (x, y, z ∈ ℝ, no NaN, no Inf)
M1-2: All faces are triangles (exactly 3 vertex indices)
M1-3: All face vertex indices are valid (0 ≤ idx < vertex_count)
M1-4: Mesh has ≥ 100 vertices and ≥ 1 face
M1-5: Bounding box fits within [-1, 1]³ (after normalization)
M1-6: UV coordinates are finite (if present)
M1-7: Material indices per face are valid (if materials present)
M1-8: No duplicate vertices (within ε = 1e-6)
```

**Note:** UV range [0, 1] is NOT required (tiling textures use UV outside [0,1]).

**Validations:**
```python
assert np.all(np.isfinite(mesh.vertices))
assert all(len(f) == 3 for f in mesh.faces)
assert all(all(0 <= i < mesh.vertex_count for i in f) for f in mesh.faces)
assert mesh.vertex_count >= 100
assert np.all(mesh.bounding_box.extents <= 2.0)
if mesh.uvs is not None:
    assert np.all(np.isfinite(mesh.uvs))
```

---

### Stage 2: Part Segmentation

**Input:** Normalized mesh

**Output:** Per-vertex semantic labels

**Label Set:**
```
{body, head, hair, clothing, accessory, hand, foot, eye, unknown}
```

**Discrete Invariants (M2):**

```
M2-1: Every vertex has exactly one label
M2-2: All labels are from the label set above
M2-3: Label count is finite and ≤ |label_set|
M2-4: "body" vertices form a connected component (largest single component)
M2-5: "head" vertices are topmost connected region by Y coordinate
M2-6: "hair" vertices are connected to "head" vertices (adjacency)
```

**Validations:**
```python
assert label_per_vertex.shape[0] == vertex_count
assert set(np.unique(labels)) ⊆ LABEL_SET
assert body_is_connected(mesh, labels)  # via graph traversal
assert head_is_topmost(mesh, labels)
```

---

### Stage 3: Skeleton Prediction

**Input:** Labeled mesh + optional reference bone template

**Output:** VRM 1.0 humanoid bone hierarchy

**Required bones (15) — MUST be present:**
```
hips, spine, head,
leftUpperArm, leftLowerArm, leftHand,
rightUpperArm, rightLowerArm, rightHand,
leftUpperLeg, leftLowerLeg, leftFoot,
rightUpperLeg, rightLowerLeg, rightFoot
```

**Optional bones (included by default):**
```
chest, upperChest, neck, leftEye, rightEye, jaw,
leftToes, rightToes,
all finger phalanges (30 bones total, 15 per side)
```

**Total target bone count: 53** (15 required + 38 optional)

**Discrete Invariants (M3):**

```
M3-1: All 15 required VRM bones are present
M3-2: Each bone has exactly one parent (except hips: parent=None)
M3-3: No cycles in parent chain (acyclic via DFS)
M3-4: All bone positions are finite (no NaN, no Inf)
M3-5: Root bone is "hips" (exactly one root)
M3-6: Bone names use VRM 1.0 camelCase convention
M3-7: Parent-child relationships match VRM 1.0 spec:
      - hips is root
      - spine.parent == hips
      - head.parent ∈ {neck, upperChest, chest, spine}
      - {side}UpperArm.parent ∈ {upperChest, chest, spine}
      - {side}LowerArm.parent == {side}UpperArm
      - {side}Hand.parent == {side}LowerArm
      - {side}UpperLeg.parent == hips
      - {side}LowerLeg.parent == {side}UpperLeg
      - {side}Foot.parent == {side}LowerLeg
M3-8: If upperChest present, chest MUST be present
M3-9: If neck present, chest or upperChest MUST be present
M3-10: Bone positions are within or adjacent to mesh bounding box
M3-11: Scale components of bone transforms are positive (VRM requirement)
```

**Validations:**
```python
required = {"hips", "spine", "head", "leftUpperArm", "leftLowerArm", "leftHand",
            "rightUpperArm", "rightLowerArm", "rightHand",
            "leftUpperLeg", "leftLowerLeg", "leftFoot",
            "rightUpperLeg", "rightLowerLeg", "rightFoot"}
assert required ⊆ set(bone.name for bone in bones)
assert exactly_one_root(bones)  # hips has no parent
assert no_cycles(bones)
assert all_bone_positions_finite(bones)
assert parent_relationships_valid(bones, VRM_PARENT_RULES)
assert upperChest_implies_chest(bones)
assert neck_implies_chest_or_upperChest(bones)
```

---

### Stage 4: Skin Weight Assignment

**Input:** Mesh + Bone positions

**Output:** Sparse weight matrix (vertex_count × bone_count)

**Discrete Invariants (M4):**

```
M4-1: All weights ≥ 0
M4-2: All weights ≤ 1
M4-3: Row sum per vertex = 1.0 ± 1e-4
M4-4: Non-zero entries per vertex ≤ 8 (GPU-friendly sparsity)
M4-5: Each bone has ≥ 1 vertex with weight > 0 (no unused bones)
M4-6: No vertex has all-zero weight (no isolated vertices)
M4-7: Weight delta between adjacent vertices < 0.5
      (relaxed for finger joints; strict threshold tunable per body region)
```

**Validations:**
```python
assert np.all(weights >= 0)
assert np.all(weights <= 1)
assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
assert np.all((weights > 0).sum(axis=1) <= 8)
assert np.all((weights > 0).any(axis=0))  # each bone used
assert np.all(weights.any(axis=1))  # each vertex has weight
# adjacency smoothness (region-dependent threshold)
for v1, v2 in mesh.adjacent_pairs():
    delta = np.abs(weights[v1] - weights[v2]).max()
    threshold = region_threshold(v1, v2)  # 0.3 body, 0.5 fingers
    assert delta < threshold
```

---

### Stage 5: Expressions (Blend Shapes)

**Input:** Mesh + Bone hierarchy

**Output:** 17 VRM 1.0 standard expressions + optional custom

**VRM 1.0 Standard Expressions (17):**
```
happy, angry, sad, relaxed, surprised,
lookUp, lookDown, lookLeft, lookRight, blink,
blinkLeft, blinkRight,
aa, ih, ou, ee, oh
```

**Neutral** is implicit (all weights = 0).

**Discrete Invariants (M5):**

```
M5-1: All 17 standard expressions are present
M5-2: Each expression has morphTarget with same vertex count as base mesh
M5-3: Expression weight range is [0.0, 1.0]
M5-4: All morphTarget vertex positions are finite (no NaN, no Inf)
M5-5: Neutral state (all weights=0) produces base mesh
M5-6: MorphTarget names are in meshes[*].extras.targetNames
M5-7: Expression→morphTarget mapping is 1:1 (each expression maps to exactly one morphTarget)
```

**Note:** M5-1 may be relaxed: at least 10 of 17 required for MVP. Quality-dependent.

**Validations:**
```python
expected = {"happy", "angry", "sad", "relaxed", "surprised",
            "lookUp", "lookDown", "lookLeft", "lookRight", "blink",
            "blinkLeft", "blinkRight", "aa", "ih", "ou", "ee", "oh"}
assert expected ⊆ set(expressions.keys())  # or len >= 10 for MVP
for expr in expressions.values():
    assert expr.morph_target.vertex_count == base_mesh.vertex_count
    assert 0.0 <= expr.preset_weight <= 1.0  # if preset
    assert np.all(np.isfinite(expr.morph_target.vertices))
```

---

### Stage 6: Spring Bones

**Input:** Labeled mesh + Bone hierarchy

**Output:** `VRMC_springBone` chains with physics parameters

**Discrete Invariants (M6):**

```
M6-1: Each spring chain has ≥ 2 collider or bone nodes
M6-2: All bone node references are valid bone names (not vertex indices)
M6-3: All collider references are valid collider indices
M6-4: No circular references in chain hierarchy
M6-5: Parameter ranges:
      stiffness ∈ [0.0, 1.0]
      dragForce ∈ [0.0, 1.0]
      hitRadius ∈ [0.0, 1.0]  (normalized to mesh scale)
      gravityPower ∈ [0.0, 10.0]
      gravityDir is a unit vector (|dir| = 1.0 ± 1e-6)
M6-6: Each spring chain has a valid root bone (parent in humanoid hierarchy)
M6-7: Collider sphere centers are within mesh bounding box
M6-8: Collider radii are positive (> 0)
```

**Validations:**
```python
for chain in spring_chains:
    assert len(chain.nodes) >= 2
    for node in chain.nodes:
        assert node.bone_name in all_bone_names
    assert 0.0 <= chain.stiffness <= 1.0
    assert 0.0 <= chain.dragForce <= 1.0
    assert 0.0 <= chain.hitRadius <= 1.0
    assert 0.0 <= chain.gravityPower <= 10.0
    assert abs(np.linalg.norm(chain.gravityDir) - 1.0) < 1e-6
    assert chain.root_bone in all_bone_names
for collider in colliders:
    assert np.all(np.isfinite(collider.center))
    assert collider.radius > 0
```

---

### Stage 7: LookAt + FirstPerson

**Input:** Bone hierarchy (eyes, head) + mesh

**Output:** `VRMC_vrm.lookAt` and `VRMC_vrm.firstPerson` configurations

**Discrete Invariants (M7):**

```
M7-1: LookAt specifies either bone-based or expression-based mode
M7-2: If bone-based: target bones are leftEye and rightEye (if present)
M7-3: If expression-based: maps to lookUp/lookDown/lookLeft/lookRight expressions
M7-4: Yaw range is finite (default ±90°, configurable)
M7-5: Pitch range is finite (default ±90°, configurable)
M7-6: FirstPerson camera offset is finite 3D vector
M7-7: FirstPerson mesh annotations: each mesh primitive has enum ∈
      {auto, both, thirdPerson, firstPerson}
```

**Validations:**
```python
lookat = vrm.lookAt
assert lookat.mode in {"bone", "expression"}
if lookat.mode == "bone":
    assert "leftEye" in bone_names and "rightEye" in bone_names
elif lookat.mode == "expression":
    assert {"lookUp", "lookDown", "lookLeft", "lookRight"} ⊆ expressions
assert np.isfinite(lookat.yaw_range)
assert np.isfinite(lookat.pitch_range)
fp = vrm.firstPerson
assert np.all(np.isfinite(fp.camera_offset))
for ann in fp.mesh_annotations:
    assert ann.type in {"auto", "both", "thirdPerson", "firstPerson"}
```

---

### Stage 8: VRM Export

**Input:** All assembled components

**Output:** Valid VRM 1.0 file (`.vrm`)

**Discrete Invariants (M8):**

```
M8-1: File is valid GLB 2.0 (passes glTF-Validator)
M8-2: VRMC_vrm extension present with specVersion "1.0"
M8-3: VRMC_vrm.humanoid present with all 15 required bones
M8-4: VRMC_vrm.meta present with required fields (see Meta Requirements below)
M8-5: VRMC_vrm.expressions present (if expressions generated)
M8-6: VRMC_vrm.lookAt present (if lookAt configured)
M8-7: VRMC_vrm.firstPerson present (if firstPerson configured)
M8-8: VRMC_springBone extension present (if spring bones generated)
M8-9: All glTF nodes referenced by humanoid bones exist
M8-10: All morphTargets referenced by expressions exist
M8-11: File size ≤ 50 MB
M8-12: Thumbnail (if present) is PNG, ≤ 1024×1024
M8-13: No `animations` or `cameras` in glTF (VRM forbids these)
M8-14: Bone transform scale components are positive (no zero scale)
```

**Validations:**
```python
gltf_validator.validate(output_path)  # throws on invalid
vrm_schema_validator.validate(output_path)  # throws on VRM schema violation
assert file_size(output_path) <= 50_000_000
if thumbnail:
    assert thumbnail.format == "PNG"
    assert thumbnail.width <= 1024
    assert thumbnail.height <= 1024
gltf = parse_glb(output_path)
assert "animations" not in gltf  # VRM forbids
assert "cameras" not in gltf  # VRM forbids
```

---

## Path B: Screenshot Decomposition

### Stage B1: Layer Decomposition

**Input:** RGBA image (H × W × 4), 1-4 views

**Output:** Per-layer masks + RGBA layer images per view

**Layer Name Set:**
```
{background, body, skin, clothing, hair_front, hair_back,
 eye, accessory, unknown}
```

**Discrete Invariants (B1):**

```
B1-1: Number of layers per view satisfies: 2 ≤ L ≤ 10
B1-2: Masks are binary ({0, 1}) per layer
B1-3: Union of all masks per view = full image domain (coverage)
B1-4: Pairwise mask intersection = ∅ (disjointness)
B1-5: Each layer has ≥ 1 pixel (no empty layers)
B1-6: All layer images have same dimensions H × W as input
B1-7: All layer images have alpha channel
B1-8: All layer names are from the layer name set above
B1-9: If multiple views, layer names are consistent across views
```

**Validations:**
```python
for view in views:
    layers = decompose(view)
    assert 2 <= len(layers) <= 10
    masks = [l.mask for l in layers]
    assert np.all(sum(masks) == 1)  # coverage + disjointness
    for l in layers:
        assert l.mask.sum() > 0
        assert l.image.shape[:2] == view.shape[:2]
        assert l.name in LAYER_NAME_SET
```

### Stage B2: Depth Estimation

**Input:** Layer images + masks (per view)

**Output:** Per-pixel depth map (H × W) per view

**Discrete Invariants (B2):**

```
B2-1: All depth values are finite
B2-2: depth.shape == image.shape[:2]
B2-3: Depth range is normalized to [0, 1] (relative depth)
B2-4: No sudden depth jumps within a single layer
      (local delta < 2 × local median, configurable threshold)
B2-5: Depth ordering respects layer semantics PER VIEW:
      - For front view: hair_front > body (hair in front)
      - For back view: hair_back > body (hair in front from back)
      - background depth < all other layers
```

**Note:** "Front/back" depth ordering is VIEW-DEPENDENT, not absolute.

**Validations:**
```python
for view in views:
    depth = estimate_depth(view.layers)
    assert np.all(np.isfinite(depth))
    assert depth.shape == view.image.shape[:2]
    assert np.all((depth >= 0) & (depth <= 1))
    # Per-view ordering check
    validate_depth_ordering(depth, view.layers, view.angle)
    # Smoothness within layer
    for layer in view.layers:
        layer_depth = depth[layer.mask]
        local_median = np.median(layer_depth)
        assert np.all(np.abs(layer_depth - local_median) < 2 * local_median + 0.01)
```

### Stage B3: Mesh Reconstruction

**Input:** Layer images + masks + depth maps (multiple views)

**Output:** Triangle mesh with per-vertex colors and UVs

**Discrete Invariants (B3):**

```
B3-1: All faces are triangles
B3-2: All vertex positions are finite
B3-3: No duplicate vertices (within ε = 1e-6)
B3-4: No degenerate faces (area > 1e-10)
B3-5: Vertex count ≤ 100,000 (decimation applied if needed)
B3-6: Mesh is 2-manifold (every edge shared by exactly 2 faces, no non-manifold edges)
B3-7: Mesh is watertight (closed surface, no holes)
B3-8: UV coordinates are finite (if present)
B3-9: Per-vertex colors are in [0, 1] (if present)
```

**Validations:**
```python
mesh = reconstruct(views)
assert np.all(np.isfinite(mesh.vertices))
assert no_duplicate_vertices(mesh, eps=1e-6)
assert all(face_area > 1e-10 for face in mesh.faces)
assert mesh.vertex_count <= 100_000
assert is_manifold(mesh)
assert is_watertight(mesh)
if mesh.uvs is not None:
    assert np.all(np.isfinite(mesh.uvs))
if mesh.colors is not None:
    assert np.all((mesh.colors >= 0) & (mesh.colors <= 1))
```

### Stage B4: Style Analysis

**Input:** Reference screenshots + reconstructed mesh

**Output:** Material parameters + texture maps

**Discrete Invariants (B4):**

```
B4-1: Color palette extracted (dominant colors, ≥ 1 per region: skin, hair, clothing)
B4-2: All color values are in [0, 1] (normalized RGB)
B4-3: Material type classified ∈ {toon, unlit, pbr}
B4-4: If toon: VRMC_materials_mtoon parameters generated
B4-5: Texture maps (if generated) are valid images (PIL-parseable)
B4-6: Texture map dimensions ≤ 2048 × 2048
B4-7: UV map covers texture [0, 1] range (no out-of-bounds sampling)
```

**Validations:**
```python
style = analyze_style(reference_images, mesh)
for region, colors in style.palette.items():
    for color in colors:
        assert all(0 <= c <= 1 for c in color)
assert style.material_type in {"toon", "unlit", "pbr"}
for tex in style.textures.values():
    img = Image.open(tex.path)
    img.verify()
    assert max(img.size) <= 2048
```

### Stage B5: Integration with Path A rig

Reuse Stages 3-8 from Path A on reconstructed mesh.

**Additional invariant:**
```
B5-1: Reconstructed mesh passes Stage 1 invariants (M1-1 through M1-8)
B5-2: All Path A stages succeed on reconstructed mesh
B5-3: Output VRM contains both geometry (from B3) and rig (from Path A stages)
```

---

## VRM Meta Requirements

```
Required:
  name:        string, non-empty, ≤ 64 chars
  version:     string, format "X.Y" (e.g., "1.0")
  authors:     array of strings, ≥ 1 author
  licenseUrl:  string (URL to license text)
  avatarPermissionType: optional enum (see VRM spec)
  allowExcessivelyViolentUsage: bool (optional)
  allowExcessivelySexualUsage: bool (optional)
  allowCommercialUsage: bool (optional)
  allowPoliticalOrReligiousUsage: bool (optional)

Optional:
  contactInformation: string
  referenceImage: base64 PNG (thumbnail)
  otherLicenseUrl: string
  thirdPartyLicenses: string
```

**Note:** `licenseUrl` is a URL, not a license name. Must point to actual license text.

---

## Acceptance Criteria

A VRM file is accepted if and only if:

```
AC1: Passes all Stage 8 invariants (M8-1 through M8-14)
     [AUTOMATED] — glTF-Validator + VRM schema

AC2: Opens in VSeeFace without errors
     [AUTOMATED] — headless VRM load test (mock VSeeFace import)
     [HUMAN]    — real VSeeFace + webcam tracking (final sign-off)

AC3: All 15 required humanoid bones present and animatable
     [AUTOMATED] — bone count + programmatic bone rotation → mesh deformation check

AC4: ≥ 10 of 17 standard expressions produce visible deformation
     [AUTOMATED] — morph delta > 1e-4 per expression (geometric)
     [AUTOMATED] — LPIPS(rendered_expression, neutral) > threshold (perceptual)
     [AUTOMATED] — CLIP(rendered_expression, expression_name) > threshold (semantic)

AC5: Spring bones (if present) oscillate when avatar moves
     [AUTOMATED] — apply impulse in headless Blender, measure oscillation
     [AUTOMATED] — damping ratio ∈ [0.3, 0.9], settling < 2s

AC6: LookAt causes eyes to follow cursor/face tracking
     [AUTOMATED] — mock face input → verify eye bone rotation within expected range
     [HUMAN]    — real webcam tracking responsiveness (final sign-off)

AC7: File size ≤ 50 MB
     [AUTOMATED] — file size check

AC8: Poly count ≤ 50,000 triangles
     [AUTOMATED] — mesh face count

AC9: Aesthetic similarity to reference (Path B only)
     [AUTOMATED] — LPIPS(rendered_avatar, reference_screenshot) < 0.3
     [AUTOMATED] — CLIP score > 0.25
     [AUTOMATED] — face landmark proportion match > 80%
     [HUMAN]    — final aesthetic sign-off (5 min)

AC10: Expression naturalness
     [AUTOMATED] — OpenFace Action Unit detection: "happy" → AU6+AU12 present
     [AUTOMATED] — L2 distance to VRoid reference expression < 0.2
     [HUMAN]    — nuanced artistic quality (edge cases)
```

**Fully automated:** AC1, AC3, AC7, AC8 + partial AC2, AC4, AC5, AC6, AC9, AC10
**Human required (minimal):** Final sign-off on AC2 (VSeeFace webcam), AC9 (aesthetic), AC10 (artistic nuance)
**Estimated human time per release:** ~10 minutes (down from ~1 hour)

---

## Automated Verification Strategy

Multi-layer verification pyramid (cheap → expensive):

```
Layer 0: Static type checking (mypy/pyright strict mode)
  Catches: wrong types, None dereference, index errors
  Cost: hours (one-time setup)

Layer 1: Runtime assertions (always on in production)
  Catches: bad data at stage boundaries in real time
  Cost: hours (built into each stage)

Layer 2: Unit tests (pytest)
  Catches: regressions in specific cases
  Cost: hours per module

Layer 3: Property-based tests (Hypothesis) ← KEY LAYER
  "For ALL possible inputs, invariants hold"
  Generates 100-1000 random meshes, checks invariants
  This is practical "discrete proof" — 95% confidence at 1% cost of Lean
  Cost: days

Layer 4: Integration tests (pipeline e2e)
  Catches: stage boundary issues
  Cost: days

Layer 5: Property-based integration (Hypothesis on full pipeline)
  Random mesh → full pipeline → verify output invariants
  Cost: days

Layer 6: Differential testing (vs VRoid reference)
  Catches: quality regression, metric drift
  Metrics: L2 bone positions, weight distribution EMD, poly count
  Cost: days

Layer 7: Perceptual metrics (ML-based) ← AUTOMATES HUMAN EVAL
  LPIPS: rendered avatar vs reference screenshot (aesthetic similarity)
  CLIP score: rendered expression vs text label (semantic correctness)
  OpenFace AU: action unit detection for expression validation
  FID: distribution-level quality (if dataset available)
  Cost: days

Layer 8: Headless VRM validation
  Load VRM programmatically, apply all operations, verify no crash
  Apply all blend shapes → verify mesh deforms
  Rotate all bones → verify mesh follows
  Mock face input → verify eye bones move (lookAt)
  Simulate spring bone physics → verify oscillation + damping
  Cost: days

Layer 9: Human (final gate) ← MINIMAL REMAINDER
  5-10 min sign-off before release
  Only: real webcam tracking, novel edge cases, artistic nuance
  Cost: minutes per release
```

**What used to require humans, now automated:**

| Aspect | Old (human) | New (automated) | Residual human |
|--------|-------------|-----------------|----------------|
| Aesthetic quality | "Looks good?" | LPIPS < 0.3, CLIP > 0.25 | Final sign-off (5 min) |
| Spring bone naturalness | "Feels right?" | Damping ∈ [0.3, 0.9], settling < 2s | Edge cases (extreme hair) |
| Expression coherence | "Happy looks happy?" | OpenFace AU detection + L2 vs VRoid | Artistic nuance |
| VSeeFace live test | Manual load + track | Headless VRM load + mock tracking | Real webcam UX |
| Similarity to reference | "Looks like character?" | LPIPS + face landmark match | Cultural/style check |
| Editability | Manual Blender edit | Programmatic re-import test | New feature edge cases |

**Lean (formal proofs) — NOT USED:**

Lean would only be justified for proving graph invariants (skeleton acyclicity, layer partition) for ALL possible inputs. But:
- Hypothesis property tests cover this at 95% confidence for 1% cost
- Runtime DFS cycle check catches any violation
- Lean formalization of Blender API is impossible

Decision: Skip Lean. Property-based testing + runtime assertions = practical "discrete proof".

---

## Out of Scope (minimal human residual)

These aspects cannot be fully automated and require human judgment in edge cases:

- Final aesthetic sign-off before release (~5 min)
- Novel artistic styles outside training data distribution
- Real webcam tracking UX (framerate, responsiveness in VSeeFace)
- Cultural/stylistic appropriateness (does avatar fit target audience?)
- New feature edge cases (untested expression combinations, extreme poses)

These are tested via manual checklist (see TEST.md → Human Evaluation Checklist).
Everything else is automated via the verification pyramid above.