"""QA for GLB / VRM schema validation and VRM thumbnail generation."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Any

import numpy as np


# ── GLB 2.0 validation ──────────────────────────────────────────────────────

GLB_MAGIC = 0x46546C67          # "glTF" in little-endian
GLB_VERSION_2 = 2
GLB_CHUNK_JSON = 0x4E4F534A     # "JSON"
GLB_CHUNK_BIN = 0x004E4942      # "BIN\0"


def validate_glb(path: str) -> bool:
    """
    Check that `path` is a structurally valid GLB 2.0 binary.

    Returns True if valid, False otherwise.
    """
    try:
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GLB_MAGIC:
                return False
            version = struct.unpack("<I", f.read(4))[0]
            if version != GLB_VERSION_2:
                return False
            length = struct.unpack("<I", f.read(4))[0]
            # Must have at least one chunk (JSON)
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_length, chunk_type = struct.unpack("<II", chunk_header)
                f.read(chunk_length)
                if chunk_type not in (GLB_CHUNK_JSON, GLB_CHUNK_BIN):
                    return False
        return True
    except Exception:
        return False


# ── VRM 1.0 schema validation ───────────────────────────────────────────────

VRM_EXTENSION_NAME = "VRMC_vrm"


def _load_glb_json(path: str) -> dict[str, Any] | None:
    """Extract and parse the JSON chunk from a GLB file."""
    try:
        with open(path, "rb") as f:
            f.read(12)  # header
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                chunk_len, chunk_type = struct.unpack("<II", header)
                data = f.read(chunk_len)
                if chunk_type == GLB_CHUNK_JSON:
                    return json.loads(data.decode("utf-8"))
        return None
    except Exception:
        return None


def _check_required(obj: dict, key: str, path: str, errors: list[str]) -> None:
    if key not in obj:
        errors.append(f"missing required field `{key}` at {path}")


def validate_vrm(path: str) -> list[str]:
    """
    Validate a VRM 1.0 file against the VRM schema.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    # 1. Must be a valid GLB
    if not validate_glb(path):
        errors.append("not a valid GLB 2.0 file")
        return errors

    gltf = _load_glb_json(path)
    if gltf is None:
        errors.append("could not parse GLB JSON chunk")
        return errors

    # 2. Check extensionsUsed / extensionsRequired
    extensions_used = set(gltf.get("extensionsUsed", []))
    if VRM_EXTENSION_NAME not in extensions_used:
        errors.append(f"extensionsUsed must contain \"{VRM_EXTENSION_NAME}\"")
        return errors

    # 3. Check extensions
    if "extensions" not in gltf:
        errors.append("missing top-level \"extensions\" object")
        return errors
    vrm_ext = gltf["extensions"].get(VRM_EXTENSION_NAME)
    if vrm_ext is None:
        errors.append(f"missing \"extensions.{VRM_EXTENSION_NAME}\"")
        return errors

    # 4. specVersion
    spec_version = vrm_ext.get("specVersion")
    if spec_version != "1.0":
        errors.append(f"specVersion must be \"1.0\", got {spec_version!r}")

    # 5. Required subfields: humanoid
    _check_required(vrm_ext, "humanoid", "extensions.VRMC_vrm", errors)

    # 6. Required subfields: meta
    _check_required(vrm_ext, "meta", "extensions.VRMC_vrm", errors)

    humanoid = vrm_ext.get("humanoid", {})
    human_bones = humanoid.get("humanBones", [])
    if len(human_bones) < 15:
        errors.append(
            f"humanoid.humanBones has {len(human_bones)} bones, need at least 15"
        )
    # Check required bone names
    required_human_bones = {
        "hips", "spine", "head",
        "leftUpperArm", "leftLowerArm", "leftHand",
        "rightUpperArm", "rightLowerArm", "rightHand",
        "leftUpperLeg", "leftLowerLeg", "leftFoot",
        "rightUpperLeg", "rightLowerLeg", "rightFoot",
    }
    found_bones = {b.get("bone") for b in human_bones if isinstance(b, dict)}
    missing = required_human_bones - found_bones
    if missing:
        errors.append(f"missing required humanoid bones: {sorted(missing)}")

    # 7. meta required fields
    meta = vrm_ext.get("meta", {})
    _check_required(meta, "name", "extensions.VRMC_vrm.meta", errors)
    _check_required(meta, "version", "extensions.VRMC_vrm.meta", errors)
    _check_required(meta, "authors", "extensions.VRMC_vrm.meta", errors)
    _check_required(meta, "licenseUrl", "extensions.VRMC_vrm.meta", errors)

    name = meta.get("name", "")
    if not name:
        errors.append("meta.name must be a non-empty string")
    if len(name) > 64:
        errors.append("meta.name must be <= 64 characters")

    authors = meta.get("authors", [])
    if not authors:
        errors.append("meta.authors must have at least one author")

    license_url = meta.get("licenseUrl", "")
    if not license_url:
        errors.append("meta.licenseUrl is required")
    elif not isinstance(license_url, str):
        errors.append("meta.licenseUrl must be a string")

    # 8. No animations (VRM forbids them — M8-13)
    if "animations" in gltf and gltf["animations"]:
        errors.append("VRM forbids animations (M8-13)")

    # 9. No cameras (VRM forbids them — M8-13)
    if "cameras" in gltf and gltf["cameras"]:
        errors.append("VRM forbids cameras (M8-13)")

    return errors


def parse_vrm(path: str) -> dict[str, Any]:
    """
    Parse VRM extension data from a GLB file.

    Returns the contents of extensions["VRMC_vrm"] as a dict.
    Raises ValueError if the file is not a valid VRM.
    """
    if not validate_glb(path):
        raise ValueError(f"{path} is not a valid GLB 2.0 file")

    gltf = _load_glb_json(path)
    if gltf is None:
        raise ValueError(f"{path}: could not parse JSON chunk")

    vrm_ext = gltf.get("extensions", {}).get(VRM_EXTENSION_NAME)
    if vrm_ext is None:
        raise ValueError(f"{path}: missing VRMC_vrm extension")

    return dict(vrm_ext)


# ── Thumbnail generation via Blender CLI ─────────────────────────────────────

BLENDER_PATH = "/mnt/b/Blender/blender.exe"


def generate_thumbnail(vrm_path: str, output_path: str, size: int = 512) -> str:
    """
    Render a VRM file to a PNG thumbnail using Blender's headless CLI.

    Parameters
    ----------
    vrm_path : str
        Path to the .vrm file.
    output_path : str
        Path where the PNG should be written.
    size : int
        Square resolution in pixels (default 512).

    Returns
    -------
    str
        The output_path on success.

    Raises
    ------
    RuntimeError
        If Blender is not found or the render fails.
    """
    if not os.path.exists(BLENDER_PATH):
        raise RuntimeError(
            f"Blender not found at {BLENDER_PATH}. "
            "Set BLENDER_PATH env var or update preview.py."
        )

    # Escape paths for Windows
    vrm_py = vrm_path.replace("\\", "\\\\")
    out_py = output_path.replace("\\", "\\\\")

    script = (
        f"import bpy; "
        f"bpy.ops.object.select_all(action='SELECT'); "
        f"bpy.ops.object.delete(); "
        f"bpy.ops.import_scene.vrm(filepath=r'{vrm_py}'); "
        f"obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None; "
        f"if obj is not None: "
        f"  bpy.context.view_layer.objects.active = obj; "
        f"  for area in bpy.context.screen.areas: "
        f"    if area.type == 'VIEW_3D': "
        f"      for region in area.regions: "
        f"        if region.type == 'WINDOW': "
        f"          override = bpy.context.copy(); "
        f"          override['area'] = area; "
        f"          override['region'] = region; "
        f"          bpy.ops.view3d.camera_to_view(override); "
        f"bpy.ops.render.render(write_still=True, "
        f"  setup=False, "
        f"  layer=bpy.context.view_layer.name, "
        f"  camera=bpy.context.scene.camera.name if bpy.context.scene.camera else ''); "
        f"bpy.data.images['Render Result'].save_render(filepath=r'{out_py}');"
    )

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
        tf.write(script)
        script_path = tf.name

    try:
        result = subprocess.run(
            [
                BLENDER_PATH,
                "--background",
                "--python-expr",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender render failed (exit {result.returncode}): {result.stderr[:1000]}"
            )
    finally:
        os.unlink(script_path)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Blender succeeded but output not found: {output_path}")

    return output_path
