# VTuber Rigger — PLAN.md
**Full implementation plan with execution order**
**Revision 2** | VRM 1.0 compliant | Blender 5.2.0 LTS

---

## Overview

Three phases, Path B is priority:

```
Phase 3: MVP — Path A (3D model → VRM)
  Validate rigging pipeline, Blender integration, VRM 1.0 export
  (~4 weeks)

Phase 4: Path B (Screenshots → VRM) — PRIORITY
  ML pipeline, layer decomposition, depth, mesh reconstruction
  (~5 weeks)

Phase 5: Verification + QA
  Automated tests, benchmarks, VSeeFace integration, profiling
  (~2 weeks, runs in parallel with 3/4)
```

**Target output:** VRM 1.0 file with `VRMC_vrm` + `VRMC_springBone` extensions.

---

## Phase 3: MVP Pipeline

### 3.1 Environment Setup

**Goal:** Verify all dependencies resolve and Blender API works.

**Critical concern:** Python 3.14 may lack torch wheels. See Step 1.

**Steps:**
```
[ ] Create venv: python3 -m venv ~/.venvs/vtuber-rigger
[ ] Activate: source ~/.venvs/vtuber-rigger/bin/activate
[ ] Upgrade pip: pip install --upgrade pip setuptools wheel
[ ] Core deps: pip install numpy scipy scikit-learn trimesh[easy]
[ ] Image: pip install pillow opencv-python-headless
[ ] Validation: pip install jsonschema pyyaml
[ ] Testing: pip install pytest pytest-cov hypothesis
[ ] Try torch: pip install torch --index-url https://download.pytorch.org/whl/cu130
    If Python 3.14 unsupported: create venv with Python 3.12 for ML stages
[ ] Verify torch CUDA: python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
[ ] Verify Blender API:
      /mnt/b/Blender/blender.exe --background --python-expr "import bpy; print(bpy.app.version)"
[ ] Create project structure (see File Structure section)
[ ] Initialize git repo: git init
[ ] Setup pyproject.toml with dependencies
```

**Blender path:** `/mnt/b/Blender/blender.exe` (Windows, WSL-callable)
**Blender version:** 5.2.0 LTS (2026-07-14 build)

**VRM 1.0 spec reference:** https://github.com/vrm-c/vrm-specification

**Verification:**
```bash
python3 -c "import numpy, scipy, sklearn, trimesh; print('Core OK')"
python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
/mnt/b/Blender/blender.exe --background --python-expr "import bpy; print(f'Blender {bpy.app.version}')"
```

**Fallback for Python 3.14 torch:**
```bash
# If torch doesn't support 3.14, create separate ML venv
python3.12 -m venv ~/.venvs/vtuber-ml  # if 3.12 available
# Or use conda/pyenv to install 3.12
```

---

### 3.2 Mesh Loader

**File:** `vtuber_rigger/mesh/loader.py`

**Purpose:** Load any supported format → normalized internal representation.

**Steps:**
```
[ ] Define MeshData dataclass: vertices, faces, normals, uvs, materials, adjacency
[ ] Format detection by extension + magic bytes:
      - .glb/.gltf: trimesh.load()
      - .vrm: trimesh.load() (VRM is glTF-based, loads as GLB)
      - .fbx: convert via Blender CLI: blender --background --python-expr "import fbx; ..."
      - .obj: trimesh.load()
[ ] Normalize: center mesh at origin, scale to fit [-1, 1]³ bounding box
[ ] Extract: vertices, faces, normals, UVs, material indices
[ ] Compute adjacency graph (scipy.sparse.csgraph from face edges)
[ ] Quality checks (M1-1 through M1-8):
      - finite vertices
      - all triangles
      - valid indices
      - ≥ 100 vertices
      - bounding box normalized
      - no duplicate vertices (merge within ε)
[ ] Export intermediate .glb for Blender step
```

**Dependencies:** numpy, scipy, trimesh
**VRM note:** trimesh loads .vrm as GLB (geometry only, ignores VRM extensions).

---

### 3.3 Part Segmentation

**File:** `vtuber_rigger/mesh/segmentation.py`

**Purpose:** Label each vertex semantically (body, head, hair, clothing, etc.).

**Steps:**
```
[ ] Compute vertex curvature via discrete Laplacian (trimesh.gradient)
[ ] Compute vertex normals
[ ] Compute symmetry plane (reflect mesh, find best mirror axis via ICP)
[ ] Cluster vertices by (curvature, position, normal) via k-means (k=6..10)
[ ] Map clusters to semantic labels via heuristics:
      - topmost cluster by Y → head
      - largest cluster → body
      - thin elongated clusters connected to head → hair
      - clusters with high curvature variance → clothing
      - remaining → accessory/unknown
[ ] Refine via graph cut (scipy.sparse.csgraph.min_cut)
[ ] Validate (M2-1 through M2-6):
      - every vertex labeled
      - labels in valid set
      - body is connected
      - head is topmost
```

**Dependencies:** scikit-learn, scipy, trimesh

---

### 3.4 Skeleton Prediction

**File:** `vtuber_rigger/rig/skeleton.py`

**Purpose:** Place VRM 1.0 humanoid bones on the mesh.

**VRM 1.0 required bones (15):**
```
hips, spine, head,
leftUpperArm, leftLowerArm, leftHand,
rightUpperArm, rightLowerArm, rightHand,
leftUpperLeg, leftLowerLeg, leftFoot,
rightUpperLeg, rightLowerLeg, rightFoot
```

**Optional bones (default include):**
```
chest, upperChest, neck, leftEye, rightEye, jaw,
leftToes, rightToes,
30 finger phalanges (3 per finger × 5 fingers × 2 sides)
```

**Steps:**
```
[ ] Define VRM 1.0 bone template (relative positions to mesh bounding box)
[ ] Scale template to mesh proportions (use segmentation labels)
[ ] Detect joint positions:
      - hips: center of body label, lowest Y of torso
      - spine: midpoint between hips and chest
      - head: center of head label
      - shoulders: top of body label, left/right extremes
      - elbows: curvature maxima on arms (between shoulder and hand)
      - wrists: end of arm clusters
      - hips joints: left/right extremes of body bottom
      - knees: curvature maxima on legs
      - ankles: end of leg clusters
      - eyes: center of eye label (if segmented)
      - fingers: subdivided from hand by proximity clustering
[ ] Build parent-child hierarchy per VRM 1.0 spec
[ ] Name bones in camelCase (VRM convention)
[ ] Validate (M3-1 through M3-11):
      - all 15 required bones present
      - no cycles
      - valid parent relationships
      - upperChest→chest implication
      - neck→chest/upperChest implication
[ ] Export as JSON for Blender consumption
```

**Dependencies:** numpy, scipy, scikit-learn

---

### 3.5 Skin Weight Assignment

**File:** `vtuber_rigger/rig/weights.py`

**Purpose:** Compute per-vertex bone influences.

**Algorithm:**
```
For each vertex v:
  1. Find k nearest bones (k=4..8, by 3D distance to bone segment)
  2. Compute RBF weights: w_i = exp(-d_i² / 2σ²), σ = mean bone distance
  3. Normalize: w_i = w_i / Σ w_j (ensures row sum = 1.0)
  4. Store as sparse matrix (scipy.sparse.csr_matrix)

For joint vertices (high curvature, between body parts):
  - Blend weights from adjacent bones with smoothing
  - Apply Laplacian smoothing to weight distribution
```

**Steps:**
```
[ ] Implement bone-to-vertex distance (point-to-segment)
[ ] Implement k-NN via scipy.spatial.cKDTree
[ ] Implement RBF kernel
[ ] Normalize rows to sum=1.0
[ ] Smooth weights via Laplacian smoothing (3 iterations)
[ ] Validate (M4-1 through M4-7):
      - all weights ≥ 0, ≤ 1
      - row sums = 1.0 ± 1e-4
      - sparsity ≤ 8 non-zero per vertex
      - each bone used
      - no isolated vertices
      - adjacency smoothness (region-dependent threshold)
[ ] Export as .npz (sparse matrix)
```

**Dependencies:** numpy, scipy, scikit-learn

---

### 3.6 Expressions (Blend Shapes)

**File:** `vtuber_rigger/rig/expressions.py`

**VRM 1.0 Standard Expressions (17):**
```
Emotion: happy, angry, sad, relaxed, surprised
Look:    lookUp, lookDown, lookLeft, lookRight, blink
Blink:   blinkLeft, blinkRight
Vowels:  aa, ih, ou, ee, oh
```

**This is the hardest stage.** Three approaches, in order of preference:

**Approach 1: Transfer from template VRM (preferred)**
```
[ ] Load template VRM with all 17 expressions (CC0 licensed)
[ ] Register template mesh to our mesh (ICP or thin-plate spline)
[ ] Transfer expression deltas via deformation transfer
[ ] Adapt deltas to our mesh topology
```

**Approach 2: Generate via bone rotations (fallback)**
```
[ ] Define bone rotation → vertex displacement for each expression
      - blink: rotate eyelid bones → eyelid vertices move down
      - happy: rotate mouth corners up → cheek vertices move
      - etc.
[ ] Compute displacement per expression
[ ] Store as morphTarget
```

**Approach 3: Manual template (MVP fallback)**
```
[ ] Precompute 17 displacement maps for a generic face mesh
[ ] Scale and apply to our mesh (approximate)
```

**Steps:**
```
[ ] Decide approach (start with Approach 3 for MVP, upgrade to 1/2 later)
[ ] Implement chosen approach
[ ] Create morphTarget per expression (same vertex count as base)
[ ] Name morphTargets in meshes[*].extras.targetNames
[ ] Validate (M5-1 through M5-7):
      - 17 expressions present (or ≥10 for MVP)
      - weight range [0, 1]
      - vertex count match
      - finite vertices
      - mapping 1:1
```

**Dependencies:** numpy, scipy (for ICP if Approach 1)

---

### 3.7 Spring Bones

**File:** `vtuber_rigger/rig/spring_bones.py`

**Purpose:** Auto-detect hair/cloth chains → VRMC_springBone config.

**Steps:**
```
[ ] Detect long edge chains (> 10 edges, thin cross-section) → hair candidates
[ ] Filter chains by:
      - connected to head label (hair) or body label (cloth)
      - length > threshold (normalized)
      - not part of main body structure
[ ] For each chain:
      - Extract node positions (bone positions along chain)
      - Compute initial stiffness (0.4-0.8 range, tunable)
      - Compute initial dragForce (0.3-0.7 range)
      - Set gravityDir = (0, -1, 0) (downward)
      - Set gravityPower = 0.5 (gentle)
      - Set hitRadius = 0.02 (normalized)
[ ] Auto-place sphere colliders:
      - Head: 1 sphere around head bone position
      - Shoulders: 2 spheres at shoulder joints
      - Waist: 1 sphere at hip level
[ ] Validate (M6-1 through M6-8):
      - chain length ≥ 2
      - valid bone references
      - parameter ranges
      - unit gravity direction
      - collider radii > 0
[ ] Export as VRMC_springBone JSON structure
```

**Dependencies:** numpy, scipy, trimesh

---

### 3.8 VRM Exporter

**File:** `vtuber_rigger/vrm/exporter.py`

**Purpose:** Assemble all components → valid VRM 1.0 `.vrm` file.

**Critical:** trimesh does NOT support skeletons/skinning/VRM extensions.
**Solution:** Use Blender for VRM export via UniVRM addon or custom glTF writer.

**Approach A: UniVRM via Blender (preferred if addon available)**
```
[ ] Check if UniVRM addon installed in Blender 5.2
[ ] If not: install UniVRM addon (download from GitHub releases)
[ ] Import mesh + armature into Blender via Python script
[ ] Apply skin weights, expressions, spring bones
[ ] Export via UniVRM exporter: bpy.ops.export_scene.vrm(filepath=...)
```

**Approach B: Custom glTF writer (fallback)**
```
[ ] Build glTF 2.0 JSON structure manually:
      - nodes (one per bone + mesh node)
      - meshes (primitives with attributes: POSITION, NORMAL, TEXCOORD, JOINTS, WEIGHTS)
      - skins (inverse bind matrices, joints array)
      - materials (VRMC_materials_mtoon)
      - accessors, bufferViews, buffers
[ ] Add VRM extensions:
      - VRMC_vrm (humanoid, meta, expressions, lookAt, firstPerson)
      - VRMC_springBone
[ ] Pack JSON header + binary buffer into GLB
[ ] Validate with glTF-Validator + VRM schema
```

**Steps:**
```
[ ] Verify UniVRM addon availability in Blender 5.2
[ ] If available: write Blender export script
[ ] If not: implement custom glTF writer (more work, more control)
[ ] Generate thumbnail: Blender render 512×512 PNG
[ ] Add meta section (name, authors, licenseUrl)
[ ] Validate output (M8-1 through M8-14)
```

**Dependencies:** Blender (via CLI), or pygltflib (if custom writer)

---

### 3.9 CLI Wrapper

**File:** `vtuber_rigger/cli.py`

**Commands:**
```bash
# Path A: Rig a 3D model
vtuber-rigger rig input.glb -o output.vrm [--quality low|medium|high]

# Path B: Reconstruct from screenshots
vtuber-rigger reconstruct --front front.png --side side.png \
  --back back.png --quarter 3q.png -o output.vrm

# Verify a VRM file
vtuber-rigger verify output.vrm

# Generate preview thumbnail
vtuber-rigger preview output.vrm -o preview.png

# Info about a VRM file
vtuber-rigger info output.vrm
```

**Steps:**
```
[ ] argparse for CLI (subcommands: rig, reconstruct, verify, preview, info)
[ ] Route to appropriate pipeline
[ ] Progress output (stage name, time elapsed, % complete)
[ ] Error handling with descriptive messages (no tracebacks for user errors)
[ ] Exit codes: 0 success, 1 user error, 2 internal error, 3 validation failure
[ ] Logging: verbose mode (-v) shows stage internals
```

---

### 3.10 QA Module

**File:** `vtuber_rigger/qa/`

**Automated checks (run on every output):**
```
[ ] geometry.py: watertight, poly count, UV validity, finite vertices
[ ] rig_check.py: bone count, cycle check, weight normalization, sparsity
[ ] expressions_check.py: count, target vertex count, weight range
[ ] spring_check.py: parameter ranges, chain connectivity, collider validity
[ ] schema.py: VRM 1.0 JSON schema validation (jsonschema)
[ ] file.py: parses as GLB, file size, no forbidden glTF items
[ ] perceptual.py: LPIPS, CLIP score, aesthetic metrics (see 3.11)
[ ] headless_vrm.py: programmatic VRM operation tests (see 5.5)
[ ] physics.py: spring bone simulation metrics (see 5.5)
```

---

### 3.11 Perceptual Metrics Module — NEW

**File:** `vtuber_rigger/qa/perceptual.py`

**Purpose:** Automate human aesthetic evaluation via ML-based perceptual metrics.

**Metrics:**
```
LPIPS (Learned Perceptual Image Patch Similarity):
  - Compare rendered avatar vs reference screenshot
  - Lower = more similar
  - Threshold: LPIPS < 0.3 = acceptable
  - Model: pretrained LPIPS (VGG/AlexNet backbone)

CLIP score:
  - Compare rendered expression vs text label
  - "happy anime face", "angry anime face", etc.
  - Higher = more semantically correct
  - Threshold: CLIP score > 0.25

OpenFace Action Unit detection:
  - Detect facial action units in rendered expression
  - "happy" → AU6 (cheek raiser) + AU12 (lip corner puller)
  - "angry" → AU4 (brow lowerer) + AU7 (lid tightener)
  - "sad" → AU1 (inner brow raiser) + AU15 (lip corner depressor)
  - Validate: expected AUs present, unexpected AUs absent

Face landmark proportion match:
  - Detect face landmarks in rendered avatar
  - Compare proportions to reference screenshot
  - Threshold: proportion match > 80%

L2 expression distance vs VRoid reference:
  - Per expression: L2(morph_delta_ours, morph_delta_vroid)
  - Threshold: L2 < 0.2 per expression
```

**Steps:**
```
[ ] Install: pip install lpips clip-by-openai openface mediapipe
[ ] Implement render_avatar(vrm_path, pose, expression) → PNG (via Blender headless)
[ ] Implement lpips_similarity(rendered, reference) → float
[ ] Implement clip_score(rendered, text_label) → float
[ ] Implement action_unit_detection(rendered) → set of AUs
[ ] Implement landmark_proportion_match(rendered, reference) → float
[ ] Integrate into QA module: run on every output
[ ] Thresholds configurable via config.yaml
```

**Dependencies:** torch, lpips, clip, openface (or mediapipe), Blender (CLI)

**Verification:** These metrics automate AC4, AC9, AC10 (see SPEC.md).

---

### 3.12 Headless VRM Validation — NEW

**File:** `vtuber_rigger/qa/headless_vrm.py`

**Purpose:** Programmatically validate VRM works without VSeeFace.

**Tests:**
```
Load test:
  - Parse VRM file
  - Load mesh, skeleton, expressions, spring bones
  - No crash = pass

Expression application:
  - For each expression (17):
    - Set weight = 1.0
    - Compute deformed mesh
    - Verify: deformed_mesh != base_mesh (visible change)
    - Verify: deformed_mesh vertices are finite

Bone rotation:
  - For each bone (15 required + optional):
    - Rotate by 10° on each axis
    - Compute deformed mesh via skin weights
    - Verify: deformed_mesh != base_mesh
    - Verify: deformed_mesh vertices are finite

LookAt simulation:
  - Mock face input: yaw=30°, pitch=15°
  - Apply to eye bones (or expressions)
  - Verify: eye bone rotation matches expected range
  - Verify: mesh around eyes deformed

Spring bone physics:
  - Apply impulse to root bone (simulate head movement)
  - Step physics simulation 60 frames
  - Measure: oscillation frequency, damping ratio, settling time
  - Validate: damping ∈ [0.3, 0.9], settling < 2s

VRM re-import:
  - Export VRM
  - Re-import VRM
  - Verify: all components preserved (bones, weights, expressions, springs)
  - Verify: no data loss in round-trip
```

**Steps:**
```
[ ] Implement VRM loader (parse GLB + VRMC_vrm extensions)
[ ] Implement expression application (morph target interpolation)
[ ] Implement bone rotation + skin deformation
[ ] Implement lookAt mock input
[ ] Implement spring bone physics simulation (Verlet integration)
[ ] Implement round-trip test (export → re-import → compare)
[ ] Integrate into QA module
```

**Dependencies:** numpy, scipy, trimesh, Blender (CLI for physics sim)

**Verification:** Automates AC2 (partial), AC3, AC4, AC5, AC6 (partial).

---

### 3.13 Physics Validation Module — NEW

**File:** `vtuber_rigger/qa/physics.py`

**Purpose:** Validate spring bone physics behavior quantitatively.

**Metrics:**
```
Oscillation frequency:
  - Apply impulse, measure oscillations per second
  - Target: 0.5-3 Hz (natural hair/cloth range)
  - Too high = stiff, too low = floppy

Damping ratio:
  - Measure amplitude decay over time
  - Target: 0.3-0.9 (underdamped, natural feel)
  - <0.3 = bouncy/unnatural, >0.9 = dead/overdamped

Settling time:
  - Time for amplitude < 1% of initial
  - Target: < 2 seconds
  - >2s = sluggish, <0.5s = stiff

Collision response:
  - Move spring chain toward collider
  - Verify: chain deflects, doesn't penetrate
  - Verify: chain recovers after passing collider
```

**Steps:**
```
[ ] Implement Verlet integration physics (numpy)
[ ] Apply standardized impulse (1m/s horizontal)
[ ] Record bone positions for 120 frames (2s at 60fps)
[ ] Compute: oscillation freq via FFT
[ ] Compute: damping ratio via exponential fit
[ ] Compute: settling time (amplitude threshold)
[ ] Implement collision test (chain vs sphere collider)
[ ] Validate against thresholds
```

**Dependencies:** numpy, scipy (FFT, curve fitting)

---

## Phase 4: Screenshot Path (PRIORITY)

### 4.1 ML Environment

**Goal:** Verify torch + ML models run on RTX 5070 12GB.

**Steps:**
```
[ ] Confirm torch CUDA works (from Phase 3.1)
[ ] Install ML deps: pip install transformers diffusers accelerate safetensors
[ ] Install image: pip install scikit-image
[ ] Download model weights:
      - See-through: clone repo, download weights from HF
      - anime-seg: pip install anime-seg, weights auto-download
      - SAM2: clone repo, download weights
      - Marigold: clone repo, download weights
[ ] Test each model loading (verify forward pass works)
[ ] Measure VRAM per model (torch.cuda.max_memory_allocated)
[ ] Plan model loading order (sequential to fit 12GB)
```

**VRAM Budget (12GB total):**
```
See-through:     ~6GB (LayerDiffuse + ControlNet)
anime-seg:       ~2GB (fallback)
SAM2:            ~3GB
Marigold:        ~3GB
Style (CLIP):    ~1GB
─────────────────────────
Sequential max:  ~6GB (one model at a time)
Parallel max:   ~12GB (risky, may OOM)
```

**Strategy:** Load models sequentially, free VRAM between stages (`del model; torch.cuda.empty_cache()`).

**See-through risk mitigation:**
```
See-through (SIGGRAPH 2026 conditionally accepted) may not have public weights.
Fallback chain:
  1. See-through (if weights available) → best quality
  2. anime-seg + SAM2 + LaMa → good quality, more steps
  3. anime-seg only → basic, no multi-layer
  4. Manual layer masking in Blender → worst case
```

---

### 4.2 Layer Decomposition

**File:** `vtuber_rigger/ml/layer_decomp.py`

**Purpose:** Decompose reference screenshots into semantic layers.

**Steps:**
```
[ ] Try See-through (if weights available):
      - Install LayerDiffuse + ControlNet
      - Run inference on each view image
      - Output: background, body, clothing, hair_front, hair_back layers
[ ] Fallback: anime-seg + SAM2:
      - anime-seg: extract character mask (foreground vs background)
      - SAM2: segment character into parts (hair, face, clothing)
      - LaMa: inpaint background behind each layer
[ ] Process all input views (front, side, back, 3q)
[ ] Output: dict {view_name: [Layer(name, mask, image), ...]}
[ ] Validate (B1-1 through B1-9):
      - coverage + disjointness (sum masks == 1)
      - no empty layers
      - valid names
      - consistent across views
```

**Dependencies:** torch, transformers, diffusers, anime-seg, SAM2

---

### 4.3 Depth Estimation

**File:** `vtuber_rigger/ml/depth.py`

**Purpose:** Recover 3D structure from layer images.

**Steps:**
```
[ ] Install Marigold (monocular depth estimation)
[ ] Estimate depth per layer image (per view)
[ ] Apply layer ordering constraint as post-processing:
      - background: lowest depth
      - body: mid depth
      - hair_front (front view): highest depth (closest to camera)
      - hair_back (front view): behind body
      - Ordering is view-dependent (see SPEC.md B2-5)
[ ] Fuse multi-view depths:
      - If 4 views provided: fuse into coherent 3D point cloud
      - Use ICP or volumetric fusion
[ ] Normalize depth to [0, 1]
[ ] Validate (B2-1 through B2-5)
```

**Dependencies:** torch, Marigold (local), scipy (for ICP)

---

### 4.4 Mesh Reconstruction

**File:** `vtuber_rigger/ml/reconstruct.py`

**Purpose:** Build 3D mesh from layers + depth maps.

**Steps:**
```
[ ] For each layer: extract silhouette polygon (OpenCV contour)
[ ] Triangulate polygon (ear clipping or Delaunay)
[ ] Extrude into 3D using depth map (vertex z = depth)
[ ] Fuse layers into coherent mesh:
      - Align via multi-view depth fusion
      - Merge overlapping regions (weighted by confidence)
[ ] Blender cleanup (via CLI):
      - Remove duplicate vertices (ε=1e-6)
      - Fill holes (boundary edges)
      - Decimate to < 50k tris (Blender: bpy.ops.mesh.decimate)
      - UV unwrap (Blender: bpy.ops.mesh.unwrap)
      - Smooth normals
[ ] Validate (B3-1 through B3-9):
      - watertight, manifold
      - vertex count ≤ 100k
      - no degenerate faces
      - finite positions
```

**Dependencies:** numpy, scipy, OpenCV, Blender (CLI)

---

### 4.5 Style Analysis

**File:** `vtuber_rigger/ml/style.py`

**Purpose:** Extract style parameters from reference screenshots.

**Steps:**
```
[ ] Load local CLIP model (or LLaVA if available locally)
[ ] Extract dominant color palette per region:
      - skin: sample from body/skin layer pixels
      - hair: sample from hair layer pixels
      - clothing: sample from clothing layer pixels
[ ] Classify eye style (size, shape, highlight pattern) via CLIP
[ ] Classify hair style (length, strands vs volume, color gradient)
[ ] Classify material type: toon vs unlit vs PBR
[ ] Map to VRMC_materials_mtoon parameters:
      - shadeColor, rimColor, outlineColor
      - shadingTooniFace, shadingShiftFactor
[ ] Generate texture maps:
      - skin texture (diffuse + shade)
      - eye texture (with highlights)
      - hair texture (with gradient)
      - Optional: use local diffusion model for texture synthesis
[ ] Apply materials to reconstructed mesh (via Blender CLI)
[ ] Validate (B4-1 through B4-7)
```

**Dependencies:** torch, transformers (CLIP), Blender (CLI)

---

### 4.6 Integration: Screenshot → VRM

**File:** `vtuber_rigger/pipeline/b_pipeline.py`

**Purpose:** Connect B1-B5 to Phase 3 rig pipeline.

**Steps:**
```
[ ] B1-B5: screenshots → reconstructed mesh + materials
[ ] Validate reconstructed mesh passes Stage 1 invariants (B5-1)
[ ] Pass to Path A pipeline:
      - Stage 1: Mesh loader (adapted for reconstructed mesh)
      - Stage 2: Part segmentation (re-run on reconstructed mesh)
      - Stage 3: Skeleton prediction
      - Stage 4: Weights
      - Stage 5: Expressions
      - Stage 6: Spring bones
      - Stage 7: LookAt + FirstPerson
      - Stage 8: VRM export
[ ] Final validation: all Stage 8 invariants
[ ] Output: .vrm file
```

---

## Phase 5: Verification

### 5.1 Automated Test Suite

**Location:** `tests/`

```
tests/
├── unit/                           # Per-module, per-invariant
│   ├── test_mesh_loader.py
│   ├── test_segmentation.py
│   ├── test_skeleton.py
│   ├── test_weights.py
│   ├── test_expressions.py
│   ├── test_spring_bones.py
│   ├── test_lookat_firstperson.py
│   ├── test_vrm_exporter.py
│   ├── test_layer_decomp.py
│   ├── test_depth.py
│   ├── test_reconstruct.py
│   └── test_style.py
├── integration/                    # Cross-stage, pipeline e2e
│   ├── test_pipeline_a_e2e.py
│   ├── test_pipeline_b_e2e.py
│   └── test_blender_integration.py
├── property/                        # Hypothesis (fuzzing-style)
│   ├── test_invariants_property.py
│   └── test_pipeline_property.py   # NEW: random mesh → full pipeline → invariants
├── perceptual/                     # NEW: ML-based aesthetic metrics
│   ├── test_lpips.py               # LPIPS avatar vs reference
│   ├── test_clip_score.py          # CLIP expression vs text label
│   ├── test_action_units.py        # OpenFace AU detection
│   └── test_landmark_match.py      # Face landmark proportion match
├── headless_vrm/                   # NEW: programmatic VRM validation
│   ├── test_vrm_load.py            # Load VRM, no crash
│   ├── test_expression_apply.py    # Apply all expressions, verify deformation
│   ├── test_bone_rotation.py       # Rotate all bones, verify mesh follows
│   ├── test_lookat_mock.py         # Mock face input, verify eye movement
│   ├── test_spring_physics.py      # Simulate spring bones, measure metrics
│   └── test_round_trip.py          # Export → re-import → compare
├── physics/                        # NEW: spring bone physics metrics
│   ├── test_oscillation.py         # Oscillation frequency 0.5-3 Hz
│   ├── test_damping.py             # Damping ratio 0.3-0.9
│   └── test_collision.py           # Collider deflection test
├── benchmarks/                     # Differential testing vs VRoid
│   └── test_vroid_comparison.py
├── fixtures/
│   └── (test data — see Test Fixtures)
└── conftest.py
```

**Each test file covers SPEC.md invariants for its stage.**

---

### 5.2 Benchmark Suite

**Purpose:** Compare output against VRoid Hub reference VRM files.

**Metrics:**
```
- Bone position L2 distance (vs VRoid mean per bone)
- Expression coverage: % of 17 expressions that produce visible deformation
- Weight distribution histogram match (Earth Mover's Distance)
- Poly count comparison
- File size comparison
- Spring bone chain count
```

**Data:**
- Download 3-5 VRoid Hub reference VRM files (CC0 license)
- Extract reference metrics (bone positions, weight distributions)
- Run our pipeline on reference inputs (screenshots of same characters)
- Compare metrics

---

### 5.3 VSeeFace / VTube Studio Integration Test

**Manual checklist (see TEST.md):**
```
[ ] Run pipeline on test GLB
[ ] Open output.vrm in VSeeFace
[ ] Verify all acceptance criteria (AC2, AC4, AC5, AC6)
[ ] Document any discrepancies
[ ] Test in VTube Studio as well (if available)
```

---

### 5.4 Resource Profiling

**Metrics per run:**
```
- Peak VRAM usage (torch.cuda.max_memory_allocated)
- Peak CPU RAM
- Stage timing breakdown
- Total pipeline time
- Disk I/O (intermediate files)
```

**Target:** Path A < 15s, Path B < 45s on RTX 5070 12GB.

---

## File Structure (Final)

```
vtuber-rigger/
├── pyproject.toml
├── README.md
├── SPEC.md
├── PLAN.md
├── TEST.md
│
├── vtuber_rigger/
│   ├── __init__.py
│   ├── cli.py                       # CLI entry point
│   │
│   ├── mesh/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Stage 1
│   │   └── segmentation.py           # Stage 2
│   │
│   ├── rig/
│   │   ├── __init__.py
│   │   ├── skeleton.py               # Stage 3
│   │   ├── weights.py                # Stage 4
│   │   ├── expressions.py            # Stage 5 (renamed from blendshapes)
│   │   ├── spring_bones.py           # Stage 6
│   │   ├── lookat_firstperson.py     # Stage 7 (NEW)
│   │   └── blender_ops.py            # Blender Python API wrapper
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── layer_decomp.py           # B1
│   │   ├── depth.py                  # B2
│   │   ├── reconstruct.py             # B3
│   │   └── style.py                  # B4
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── a_pipeline.py             # Path A orchestrator
│   │   └── b_pipeline.py             # Path B orchestrator (NEW)
│   │
│   ├── vrm/
│   │   ├── __init__.py
│   │   ├── spec.py                   # VRM 1.0 JSON helpers
│   │   ├── bone_mapping.py           # 15 required + optional bones
│   │   ├── expression_lib.py         # 17 standard expressions (renamed)
│   │   ├── material_templates.py     # VRMC_materials_mtoon presets
│   │   └── exporter.py               # Stage 8
│   │
│   ├── qa/
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   ├── rig_check.py
│   │   ├── expressions_check.py
│   │   ├── spring_check.py
│   │   ├── schema.py                 # VRM 1.0 JSON schema validation
│   │   ├── preview.py                # Thumbnail generation
│   │   ├── perceptual.py             # NEW: LPIPS, CLIP, OpenFace AU metrics
│   │   ├── headless_vrm.py           # NEW: programmatic VRM operation tests
│   │   └── physics.py                # NEW: spring bone simulation metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── blender.py                # Blender CLI wrapper
│
├── scripts/
│   ├── install_deps.py
│   ├── run_blender.py                # Blender script runner
│   ├── download_vroid_refs.py        # Download benchmark fixtures
│   └── generate_test_mesh.py         # Generate procedural test mesh
│
└── tests/
    ├── unit/
    ├── integration/
    ├── property/
    ├── benchmarks/
    ├── fixtures/
    └── conftest.py
```

---

## Execution Order

```
Week 1:   Phase 3.1 (env setup) + 3.2 (mesh loader) + start 3.3 (segmentation)
Week 2:   Phase 3.3 (finish) + 3.4 (skeleton) + 3.5 (weights)
Week 3:   Phase 3.6 (expressions — hardest) + 3.7 (spring bones)
Week 4:   Phase 3.8 (VRM exporter) + 3.9 (CLI) + 3.10 (QA core)
          → MVP working: GLB → VRM

Week 5:   Phase 3.11 (perceptual metrics) + 3.12 (headless VRM) + 3.13 (physics)
          + Phase 5.1 (test suite: unit + property)
          → Automated verification replaces 90% of human eval

Week 6:   Phase 4.1 (ML env) + 4.2 (layer decomp)
Week 7:   Phase 4.3 (depth) + 4.4 (mesh reconstruction)
Week 8:   Phase 4.5 (style) + 4.6 (integration)
          → Path B working: screenshots → VRM

Week 9:   Phase 5.2 (benchmarks vs VRoid) + 5.3 (VSeeFace final sign-off)
          + Phase 5.4 (profiling) + optimization + documentation
```

**MVP (Phase 3 core):** ~4 weeks
**Automated QA (Phase 3.11-3.13 + 5.1):** ~1 week (parallel with Phase 4)
**Path B (Phase 4):** ~3 weeks
**Final verification (Phase 5):** ~1 week (overlaps with 4)
**Total:** ~9 weeks

**Key milestone:** After Week 5, human evaluation reduced from ~1 hour to ~10 min per release.

---

## Key Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Python 3.14 + torch incompatibility | High | Create separate venv with Python 3.12 for ML stages |
| See-through weights not public | High | Fallback to anime-seg + SAM2 (see 4.1) |
| trimesh can't handle skeletons/skinning | High | Use Blender for export (Approach A in 3.8) |
| UniVRM addon not available for Blender 5.2 | Medium | Custom glTF writer (Approach B in 3.8) |
| Expression generation is hard | High | Start with Approach 3 (template), upgrade later |
| Mesh from 2D is low quality | High | Treat as stylized approximation; human eval gate |
| 12GB VRAM insufficient for all ML models | Medium | Sequential model loading, free VRAM between stages |
| VRM 1.0 spec complexity | Medium | Reference official spec, use jsonschema validation |
| Blender CLI slow startup (~5s per call) | Low | Batch operations in single Blender script call |

---

## Dependencies Summary

**Core (Python):**
```txt
numpy, scipy, scikit-learn, trimesh[easy]
pillow, opencv-python-headless
jsonschema, pyyaml
pytest, pytest-cov, hypothesis
```

**ML (Python, separate venv if 3.14 incompatible):**
```txt
torch (CUDA 13 / cu130)
transformers, diffusers, accelerate, safetensors
anime-seg
scikit-image
# Plus model weights downloaded separately
```

**External tools:**
```txt
Blender 5.2.0 LTS (/mnt/b/Blender/blender.exe)
UniVRM addon (for Blender, if available)
glTF-Validator (for VRM file validation)
```

**Models (downloaded once at install):**
```
See-through weights (if available)
SAM2 weights
Marigold weights
anime-seg weights (via pip)
CLIP model (via transformers)
```