"""Integration tests for Path A pipeline (GLB → VRM)."""

import pytest
import numpy as np
import trimesh
import tempfile
import os
from pathlib import Path

from vtuber_rigger.pipeline.a_pipeline import run_pipeline_a


@pytest.fixture
def test_glb_path(tmp_path):
    """Create a test GLB file from an icosphere."""
    sphere = trimesh.creation.icosphere(subdivisions=3)
    # Elongate to create a humanoid-ish shape
    sphere.vertices[:, 1] *= 1.5
    glb_path = tmp_path / "test_avatar.glb"
    sphere.export(str(glb_path), file_type="glb")
    return str(glb_path)


@pytest.fixture
def output_vrm_path(tmp_path):
    """Output VRM path."""
    return str(tmp_path / "output.vrm")


class TestPipelineAEndToEnd:

    def test_path_a_glb_to_vrm(self, test_glb_path, output_vrm_path):
        """Full Path A: GLB → VRM, all stages pass."""
        result = run_pipeline_a(test_glb_path, output_vrm_path)

        assert result.exit_code == 0, f"Pipeline failed: {result.errors}"
        assert os.path.exists(output_vrm_path), "Output file not created"

    def test_output_is_valid_vrm(self, test_glb_path, output_vrm_path):
        """Output passes VRM validation."""
        from vtuber_rigger.vrm.exporter import validate_vrm

        run_pipeline_a(test_glb_path, output_vrm_path)
        errors = validate_vrm(output_vrm_path)
        assert not errors, f"VRM validation errors: {errors}"

    def test_output_file_size_reasonable(self, test_glb_path, output_vrm_path):
        """Output file size < 50 MB."""
        run_pipeline_a(test_glb_path, output_vrm_path)
        size_mb = os.path.getsize(output_vrm_path) / 1e6
        assert size_mb < 50, f"File too large: {size_mb:.1f} MB"

    def test_stage_times_recorded(self, test_glb_path, output_vrm_path):
        """Pipeline records timing for each stage."""
        result = run_pipeline_a(test_glb_path, output_vrm_path)
        assert hasattr(result, "stage_times")
        assert len(result.stage_times) > 0
        assert all(t >= 0 for t in result.stage_times.values())

    def test_qa_checks_included(self, test_glb_path, output_vrm_path):
        """Pipeline result includes QA check output."""
        result = run_pipeline_a(test_glb_path, output_vrm_path)
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        # If exit_code == 0, no critical errors
        if result.exit_code == 0:
            assert not result.errors or all("M4-7" in e for e in result.errors)


class TestPipelineAErrorHandling:

    def test_nonexistent_input_returns_error(self, tmp_path):
        """Non-existent input file returns non-zero exit code."""
        output = str(tmp_path / "out.vrm")
        result = run_pipeline_a("/nonexistent/file.glb", output)
        assert result.exit_code != 0
        assert len(result.errors) > 0

    def test_invalid_input_returns_error(self, tmp_path):
        """Invalid file content returns non-zero exit code."""
        bad_path = tmp_path / "bad.glb"
        bad_path.write_bytes(b"not a valid glb file")
        output = str(tmp_path / "out.vrm")
        result = run_pipeline_a(str(bad_path), output)
        assert result.exit_code != 0


class TestCLI:

    def test_cli_rig_command(self, test_glb_path, output_vrm_path):
        """CLI 'rig' command produces VRM."""
        from vtuber_rigger.cli import main
        import sys

        old_argv = sys.argv
        sys.argv = ["vtuber-rigger", "rig", test_glb_path, "-o", output_vrm_path]
        try:
            exit_code = main()
            assert exit_code == 0
            assert os.path.exists(output_vrm_path)
        finally:
            sys.argv = old_argv

    def test_cli_verify_command(self, test_glb_path, output_vrm_path):
        """CLI 'verify' command validates VRM."""
        from vtuber_rigger.cli import main
        import sys

        # First generate a VRM
        run_pipeline_a(test_glb_path, output_vrm_path)

        old_argv = sys.argv
        sys.argv = ["vtuber-rigger", "verify", output_vrm_path]
        try:
            exit_code = main()
            # Should return 0 if valid
            assert exit_code in (0, 3)  # 0=valid, 3=validation errors
        finally:
            sys.argv = old_argv