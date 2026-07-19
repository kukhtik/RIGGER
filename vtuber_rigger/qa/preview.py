"""VRM thumbnail generation via Blender CLI."""

from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path

BLENDER = "/mnt/b/Blender/blender.exe"


def generate_thumbnail(
    vrm_path: str,
    output_path: str,
    size: int = 512,
) -> str:
    """Render a thumbnail PNG from a VRM file using Blender.

    Parameters
    ----------
    vrm_path : str
        Path to the .vrm file.
    output_path : str
        Path to save the thumbnail PNG.
    size : int
        Output image size in pixels (square). Default 512.

    Returns
    -------
    str
        Path to the generated PNG file.

    Raises
    ------
    FileNotFoundError
        If VRM file or Blender executable not found.
    RuntimeError
        If Blender fails to render.
    """
    if not os.path.exists(vrm_path):
        raise FileNotFoundError(f"VRM file not found: {vrm_path}")
    if not os.path.exists(BLENDER):
        raise FileNotFoundError(f"Blender not found: {BLENDER}")

    # Use forward slashes for paths (works on Windows too)
    vrm_fwd = vrm_path.replace("\\", "/")
    out_fwd = output_path.replace("\\", "/")

    # Write a Python script for Blender to execute
    script = f"""
import bpy
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import VRM (try different importers)
try:
    bpy.ops.import_vrm.plotting_vrm(filepath=r'{vrm_fwd}')
except:
    try:
        bpy.ops.import_scene.gltf(filepath=r'{vrm_fwd}')
    except Exception as e:
        print(f"Import error: {{e}}")

# Find imported object
obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
if obj is None:
    # Try active object
    obj = bpy.context.active_object

if obj is not None:
    # Set object location to origin
    obj.location = (0, 0, 0)

    # Add camera
    bpy.ops.object.camera_add(location=(0, -2, 1), rotation=(1.2, 0, 0))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam

    # Add light
    bpy.ops.object.light_add(type='SUN', location=(0, 2, 3))
    light = bpy.context.active_object
    light.data.energy = 3.0

    # Set render settings
    scene = bpy.context.scene
    scene.render.resolution_x = {size}
    scene.render.resolution_y = {size}
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = r'{out_fwd}'
    scene.render.engine = 'BLENDER_EEVEE'

    # Render
    bpy.ops.render.render(write_still=True)
    print("RENDER_OK")
else:
    print("NO_OBJECT_FOUND")
"""

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # Run Blender with script
        cmd = [BLENDER, "--background", "--python", script_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if "RENDER_OK" not in result.stdout and not os.path.exists(output_path):
            raise RuntimeError(
                f"Blender render failed: {result.stderr[:500]}"
            )

    finally:
        os.unlink(script_path)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Thumbnail not generated: {output_path}")

    return output_path