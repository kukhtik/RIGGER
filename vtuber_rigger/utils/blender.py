"""Blender CLI wrapper — batch all operations in one call to avoid startup overhead."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

BLENDER_PATH = "/mnt/b/Blender/blender.exe"


def _blender_available() -> bool:
    return Path(BLENDER_PATH).exists()


def run_blender_script(script_path: str, workdir: str = None) -> subprocess.CompletedProcess:
    """
    Run a Blender Python script file via CLI.

    Parameters
    ----------
    script_path : str
        Absolute path to the .py script.
    workdir : str, optional
        Working directory for the subprocess.

    Returns
    -------
    subprocess.CompletedProcess
    """
    if not _blender_available():
        raise RuntimeError(f"Blender not found at {BLENDER_PATH}")

    cmd = [BLENDER_PATH, "--background", "--python", script_path]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=workdir,
    )


def run_blender_expr(python_expr: str) -> subprocess.CompletedProcess:
    """
    Run an inline Python expression via Blender --python-expr.

    Parameters
    ----------
    python_expr : str
        One-liner Python code to execute inside Blender.

    Returns
    -------
    subprocess.CompletedProcess
    """
    if not _blender_available():
        raise RuntimeError(f"Blender not found at {BLENDER_PATH}")

    cmd = [BLENDER_PATH, "--background", "--python-expr", python_expr]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def run_blender_script_batch(script_blocks: list[str], output_path: str) -> subprocess.CompletedProcess:
    """
    Write multiple Python blocks to a single temp script and run Blender once.

    Each block is joined with a blank line. Use this to batch Blender operations
    and avoid paying the ~5 s startup cost multiple times.

    Parameters
    ----------
    script_blocks : list[str]
        Ordered list of Python source snippets to concatenate.
    output_path : str
        Path where the temporary script will be written (auto-deleted after run).

    Returns
    -------
    subprocess.CompletedProcess
    """
    if not _blender_available():
        raise RuntimeError(f"Blender not found at {BLENDER_PATH}")

    combined = "\n\n".join(script_blocks)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(combined)
        script_path = f.name

    try:
        return run_blender_script(script_path, workdir=os.path.dirname(output_path) or None)
    finally:
        Path(script_path).unlink(missing_ok=True)
