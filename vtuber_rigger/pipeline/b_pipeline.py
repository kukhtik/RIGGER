"""Path B: Screenshots → VRM pipeline orchestrator (ML-based reconstruction)."""

from __future__ import annotations
import time
from typing import Optional

from vtuber_rigger.interfaces import VRMMeta
from vtuber_rigger.pipeline.a_pipeline import PipelineResult

# ML Stage B1-B4
from vtuber_rigger.ml.layer_decomp import decompose_layers
from vtuber_rigger.ml.depth import estimate_depth
from vtuber_rigger.ml.reconstruct import reconstruct_mesh
from vtuber_rigger.ml.style import analyze_style

# Rigging Stage (Shared with Path A)
from vtuber_rigger.rig.skeleton import predict_skeleton
from vtuber_rigger.rig.weights import compute_weights
from vtuber_rigger.rig.expressions import create_expressions
from vtuber_rigger.rig.spring_bones import detect_spring_bones
from vtuber_rigger.rig.lookat_firstperson import create_lookat, create_firstperson
from vtuber_rigger.vrm.exporter import export_vrm, validate_vrm


def run_pipeline_b(
    screenshots: dict,
    output_path: str,
    meta: Optional[VRMMeta] = None,
) -> PipelineResult:
    """
    Run the Path B reconstruction pipeline: screenshots → VRM file.

    Parameters
    ----------
    screenshots : dict
        Dictionary of view name → image file path.
        Expected keys: 'front', 'side', 'back', 'quarter' (all optional but
        at least one is required).
    output_path : str
        Destination .vrm file.
    meta : VRMMeta, optional
        VRM metadata. Uses defaults if not provided.

    Returns
    -------
    PipelineResult
    """
    errors = []
    warnings = []
    stage_times = {}
    start_time = time.time()

    if not screenshots:
        return PipelineResult(
            exit_code=1,
            output_path=output_path,
            errors=["No screenshots provided for reconstruction."],
            warnings=warnings,
            stage_times=stage_times,
        )

    if len(screenshots) < 2:
        warnings.append("Only one view provided; reconstruction quality will be reduced.")

    try:
        # --- Stage B1: Decompose Layers ---
        s_start = time.time()
        decomposed_views = {}
        for view, path in screenshots.items():
            # decompose_layers returns a list of Layer objects
            decomposed_views[view] = decompose_layers(path)
        stage_times["decompose_layers"] = time.time() - s_start

        # --- Stage B2: Estimate Depth ---
        s_start = time.time()
        depth_maps = {}
        for view, path in screenshots.items():
            # Pass layer masks to depth estimation for better consistency (B2-5)
            layer_masks = {layer.name: layer.mask for layer in decomposed_views[view]}
            depth_maps[view] = estimate_depth(path, layer_masks=layer_masks)
        stage_times["estimate_depth"] = time.time() - s_start

        # --- Stage B3: Reconstruct Mesh ---
        s_start = time.time()
        # reconstruct_mesh expects a list of masks and a map of layer_idx -> depth_map
        # We'll use the 'front' view as primary, or the first available view.
        primary_view = next(iter(screenshots))
        
        # Prepare layers list (masks only) and depth map for the primary view
        layers_masks = [layer.mask for layer in decomposed_views[primary_view]]
        depth_map_primary = depth_maps[primary_view]
        
        # Create the expected map of layer_idx -> depth_map (B3 requires this)
        # In our simple case, one depth map shared across layers or per-layer logic
        # Since estimate_depth produces one HxW map, we map every layer to it.
        depth_maps_per_layer = {i: depth_map_primary for i in range(len(layers_masks))}
        
        mesh = reconstruct_mesh(layers_masks, depth_maps_per_layer)
        stage_times["reconstruct_mesh"] = time.time() - s_start

        # --- Stage B4: Analyze Style ---
        s_start = time.time()
        materials = analyze_style(screenshots)
        stage_times["analyze_style"] = time.time() - s_start

        # --- Path A Reuse Stages ---
        # Rigging starts here
        s_start = time.time()
        skeleton = predict_skeleton(mesh, meta)
        stage_times["predict_skeleton"] = time.time() - s_start

        s_start = time.time()
        weights = compute_weights(mesh, skeleton)
        stage_times["compute_weights"] = time.time() - s_start

        s_start = time.time()
        expressions = create_expressions(mesh, skeleton)
        stage_times["create_expressions"] = time.time() - s_start

        s_start = time.time()
        spring_bones = detect_spring_bones(mesh, skeleton)
        stage_times["detect_spring_bones"] = time.time() - s_start

        s_start = time.time()
        lookat = create_lookat(skeleton)
        firstperson = create_firstperson(skeleton)
        stage_times["create_lookat_firstperson"] = time.time() - s_start

        # Export
        s_start = time.time()
        export_vrm(
            mesh,
            skeleton,
            weights,
            expressions,
            spring_bones,
            lookat,
            firstperson,
            materials,
            output_path,
            meta,
        )
        validate_vrm(output_path)
        stage_times["export_vrm"] = time.time() - s_start

        return PipelineResult(
            exit_code=0,
            output_path=output_path,
            errors=errors,
            warnings=warnings,
            stage_times=stage_times,
        )

    except Exception as e:
        errors.append(f"Pipeline B failure: {str(e)}")
        return PipelineResult(
            exit_code=1,
            output_path=output_path,
            errors=errors,
            warnings=warnings,
            stage_times=stage_times,
        )
