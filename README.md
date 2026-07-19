# VTuber Rigger

Self-hosted AI-powered auto-rigger for VTuber avatars. Two paths:

- **Path A**: 3D model (GLB/GLTF/VRM/OBJ) → VRM 1.0
- **Path B**: Screenshots (front/side/back views) → VRM 1.0

No paid APIs. Local models only. RTX 5070 recommended for ML inference.

## Quick Start

```bash
# Install
pip install -e .

# Path A: GLB → VRM (0.4s)
vtuber-rigger rig input.glb -o output.vrm

# Path B: Screenshots → VRM
vtuber-rigger reconstruct --front front.png --side side.png --back back.png -o output.vrm

# Verify output
vtuber-rigger verify output.vrm

# Preview in Blender
vtuber-rigger preview output.vrm
```

## Requirements

- Python 3.11+
- Blender 5.2 LTS (optional, for preview)
- torch 2.13+ (optional, for ML-enhanced quality)
- VRM 1.0 compliant target

## Project Structure

```
vtuber_rigger/
├── cli.py              # Command-line interface
├── interfaces.py       # Shared dataclasses (MeshData, Skeleton, VRMMeta...)
├── mesh/
│   ├── loader.py      # Load GLB/GLTF/VRM/OBJ
│   └── segmentation.py # K-Means + curvature body part segmentation
├── rig/
│   ├── skeleton.py    # 53-bone VRM skeleton prediction
│   ├── weights.py     # RBF + Laplacian skin weights
│   ├── expressions.py  # 17 VRM expressions
│   ├── spring_bones.py # Hair/clothing spring detection
│   └── lookat_firstperson.py
├── vrm/
│   └── exporter.py    # GLB + VRM 1.0 extensions export
├── ml/
│   ├── layer_decomp.py # Semantic layer decomposition (B1)
│   ├── depth.py       # Monocular depth estimation (B2)
│   ├── reconstruct.py # Mesh reconstruction from depth (B3)
│   └── style.py       # Color palette + material analysis (B4)
├── pipeline/
│   ├── a_pipeline.py  # Path A end-to-end
│   └── b_pipeline.py  # Path B end-to-end
└── qa/
    ├── geometry.py     # Mesh validity checks
    ├── rig_check.py   # Bone hierarchy + constraints
    ├── schema.py      # GLB/VRM schema validation
    ├── perceptual.py  # LPIPS + CLIP perceptual metrics
    ├── headless_vrm.py # Headless VRM expression testing
    └── physics.py     # Spring bone physics validation
```

## VRM 1.0 Compliance

- **15 required bones**: hips, spine, chest, neck, head, arms (L/R), hands (L/R), legs (L/R), feet (L/R), toes (L/R)
- **17 standard expressions**: happy, angry, sad, relaxed, surprised, look at (left/right/up/down), blink (L/R), aa, ih, ou, ee, oh
- **Spring bones**: auto-detected for hair/clothing
- **LookAt + FirstPerson**: built-in

## Testing

```bash
# All tests
python3 -m pytest tests/ -v

# Coverage
python3 -m pytest tests/ --cov=vtuber_rigger --cov-report=term-missing
```

## Architecture

Invariant-driven design. Every stage enforces formal contracts:

- **B1-1..B1-9**: Layer decomposition invariants
- **B2-1..B2-5**: Depth estimation invariants
- **B3-1..B3-9**: Mesh reconstruction invariants
- **Stage 1-8**: Full GLB→VRM pipeline invariants

See `SPEC.md` for formal specification.

## Dependencies

```
Core:       numpy, scipy, scikit-learn, trimesh, pygltflib, opencv-python-headless, pillow, jsonschema, pyyaml
ML:         torch 2.13+, torchvision 0.28+, lpips, transformers
Optional:   Blender 5.2 LTS (for preview)
Testing:    pytest, hypothesis, pytest-cov
```

## License

MIT
