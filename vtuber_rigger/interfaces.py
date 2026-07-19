# Core data structures shared across stages
# Sub-agents MUST use these interfaces to stay compatible

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─── Stage 1: Mesh ─────────────────────────────────────────────

@dataclass
class MeshData:
    """Normalized triangle mesh. Output of Stage 1 (mesh/loader.py)."""
    vertices: np.ndarray          # (N, 3) float32, finite, bounded [-1,1]
    faces: np.ndarray             # (M, 3) int32, valid indices
    normals: Optional[np.ndarray] = None   # (N, 3) float32
    uvs: Optional[np.ndarray] = None      # (N, 2) float32, finite (no range constraint)
    material_indices: Optional[np.ndarray] = None  # (M,) int32
    vertex_colors: Optional[np.ndarray] = None     # (N, 4) float32 [0,1]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (min, max) corners."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


# ─── Stage 2: Segmentation ─────────────────────────────────────

LABEL_SET = frozenset({
    "body", "head", "hair", "clothing", "accessory",
    "hand", "foot", "eye", "unknown"
})

@dataclass
class LabeledMesh(MeshData):
    """Mesh + per-vertex semantic labels. Output of Stage 2 (mesh/segmentation.py)."""
    labels: np.ndarray = field(default_factory=lambda: np.array([]))  # (N,) str


# ─── Stage 3: Skeleton ─────────────────────────────────────────

REQUIRED_BONES = frozenset({
    "hips", "spine", "head",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
})

# VRM 1.0 parent rules: bone_name → required_parent (None = multiple options)
VRM_PARENT_RULES: dict[str, Optional[str]] = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "upperChest": "chest",
    "neck": None,  # chest or upperChest
    "head": None,  # neck, upperChest, chest, or spine
    "leftEye": "head",
    "rightEye": "head",
    "jaw": "head",
    "leftUpperArm": None,  # upperChest, chest, or spine
    "leftLowerArm": "leftUpperArm",
    "leftHand": "leftLowerArm",
    "leftUpperLeg": "hips",
    "leftLowerLeg": "leftUpperLeg",
    "leftFoot": "leftLowerLeg",
    "leftToes": "leftFoot",
    "rightUpperArm": None,
    "rightLowerArm": "rightUpperArm",
    "rightHand": "rightLowerArm",
    "rightUpperLeg": "hips",
    "rightLowerLeg": "rightUpperLeg",
    "rightFoot": "rightLowerLeg",
    "rightToes": "rightFoot",
}

@dataclass
class Bone:
    name: str               # VRM camelCase
    position: np.ndarray    # (3,) float32
    parent: Optional[str] = None  # parent bone name
    rotation: Optional[np.ndarray] = None  # (3,) euler radians


@dataclass
class Skeleton:
    bones: list[Bone]

    @property
    def bone_names(self) -> set[str]:
        return {b.name for b in self.bones}

    @property
    def bone_map(self) -> dict[str, Bone]:
        return {b.name: b for b in self.bones}

    def get_bone(self, name: str) -> Optional[Bone]:
        return self.bone_map.get(name)


# ─── Stage 4: Weights ──────────────────────────────────────────

from scipy.sparse import csr_matrix

@dataclass
class SkinWeights:
    """Sparse weight matrix. Output of Stage 4 (rig/weights.py)."""
    matrix: csr_matrix       # (vertex_count, bone_count) float32
    bone_names: list[str]     # column index → bone name

    def get_vertex_weights(self, vertex_idx: int) -> dict[str, float]:
        row = self.matrix.getrow(vertex_idx).toarray().ravel()
        return {name: w for name, w in zip(self.bone_names, row) if w > 0}


# ─── Stage 5: Expressions ──────────────────────────────────────

VRM_STANDARD_EXPRESSIONS = frozenset({
    "happy", "angry", "sad", "relaxed", "surprised",
    "lookUp", "lookDown", "lookLeft", "lookRight", "blink",
    "blinkLeft", "blinkRight",
    "aa", "ih", "ou", "ee", "oh",
})

@dataclass
class MorphTarget:
    name: str
    vertices: np.ndarray     # (N, 3) delta or absolute? — absolute positions
    vertex_count: int

@dataclass
class Expression:
    name: str                # one of VRM_STANDARD_EXPRESSIONS or custom_*
    morph_target: MorphTarget
    morph_target_index: int  # index into glTF morphTargets array
    preset_weight: float = 1.0  # default weight for this expression


# ─── Stage 6: Spring Bones ─────────────────────────────────────

@dataclass
class SpringNode:
    bone_name: str
    position: np.ndarray     # (3,) float32

@dataclass
class SpringChain:
    nodes: list[SpringNode]
    root_bone: str
    stiffness: float = 0.5       # [0, 1]
    dragForce: float = 0.5       # [0, 1]
    hitRadius: float = 0.02      # [0, 1]
    gravityPower: float = 0.5   # [0, 10]
    gravityDir: np.ndarray = field(default_factory=lambda: np.array([0, -1, 0], dtype=np.float32))

@dataclass
class Collider:
    center: np.ndarray       # (3,) float32
    radius: float            # > 0


# ─── Stage 7: LookAt + FirstPerson ─────────────────────────────

@dataclass
class LookAt:
    mode: str                 # "bone" or "expression"
    yaw_range: float = 1.5708  # radians (~90°)
    pitch_range: float = 1.5708

@dataclass
class FirstPerson:
    camera_offset: np.ndarray  # (3,) float32
    mesh_annotations: list[dict]  # [{"node": idx, "type": "auto"|"both"|"thirdPerson"|"firstPerson"}]


# ─── Stage 8: Meta ─────────────────────────────────────────────

@dataclass
class VRMMeta:
    name: str
    version: str = "1.0"
    authors: list[str] = field(default_factory=lambda: ["Unknown"])
    licenseUrl: str = "https://creativecommons.org/publicdomain/zero/1.0/"
    contactInformation: str = ""
    referenceImage: Optional[bytes] = None  # PNG bytes