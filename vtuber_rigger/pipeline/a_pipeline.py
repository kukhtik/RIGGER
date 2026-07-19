"""Path A: 3D model (GLB/GLTF/VRM/FBX/OBJ) → VRM pipeline orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vtuber_rigger.interfaces import (
    MeshData,
    LabeledMesh,
    Skeleton,
    SkinWeights,
    Expression,
    SpringChain,
    Collider,
    LookAt,
    FirstPerson,
    VRMMeta,
)
from vtuber_rigger.mesh.loader import load_mesh
from vtuber_rigger.mesh.segmentation import segment
from vtuber_rigger.rig.skeleton import predict_skeleton
from vtuber_rigger.rig.weights import compute_weights
from vtuber_rigger.rig.expressions import create_expressions
from vtuber_rigger.rig.spring_bones import detect_spring_bones
from vtuber_rigger.rig.lookat_firstperson import create_lookat, create_firstperson
from vtuber_rigger.vrm.exporter import export_vrm, validate_vrm
from vtuber_rigger.qa.geometry import check_geometry
from vtuber_rigger.qa.rig_check import check_rig

# ─── Result dataclass ────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Outcome of a pipeline run."""

    exit_code: int          # 0 = success, non-zero = error
    output_path: str        # path to produced VRM (empty if failed)
    errors: list[str]       # pipeline-level errors
    warnings: list[str]     # QA warnings (non-fatal)
    stage_times: dict[str, float]  # seconds per stage


# ─── Default meta ───────────────────────────────────────────────────────────


_DEFAULT_META = VRMMeta(
    name="VTuber Avatar",
    authors=["VTuber Rigger"],
    licenseUrl="https://creativecommons.org/publicdomain/zero/1.0/",
)


# ─── Pipeline ────────────────────────────────────────────────────────────────


def run_pipeline_a(
    input_path: str,
    output_path: str,
    meta: Optional[VRMMeta] = None,
) -> PipelineResult:
    """
    Run the full Path A rigging pipeline on a 3D model file.

    Stages
    ------
    1. load_mesh        → MeshData
    2. segment           → LabeledMesh
    3. predict_skeleton  → Skeleton
    4. compute_weights   → SkinWeights
    5. create_expressions→ list[Expression]
    6. detect_spring_bones → list[SpringChain], list[Collider]
    7. create_lookat     → LookAt
    8. create_firstperson → FirstPerson
    9. export_vrm        → writes output_path

    QA checks (geometry + rig) are run on the final mesh and rig.

    Parameters
    ----------
    input_path : str
        Path to GLB, GLTF, VRM, FBX, or OBJ file.
    output_path : str
        Destination .vrm file.
    meta : VRMMeta, optional
        VRM metadata.  Uses defaults if not provided.

    Returns
    -------
    PipelineResult
    """
    meta = meta or _DEFAULT_META
    stage_times: dict[str, float] = {}
    errors: list[str] = []
    warnings: list[str] = []

    def _tick(label: str):
        return time.perf_counter(), label

    def _tock(start: float, label: str):
        stage_times[label] = time.perf_counter() - start

    t0 = time.perf_counter()
    stage = "load_mesh"
    try:
        mesh: MeshData = load_mesh(input_path)
    except Exception as exc:
        return PipelineResult(
            exit_code=2,
            output_path="",
            errors=[f"{stage}: {exc}"],
            warnings=[],
            stage_times={stage: time.perf_counter() - t0},
        )
    _tock(t0, stage)

    # ── Stage 2: Segmentation ────────────────────────────────────────
    t1 = time.perf_counter()
    stage = "segment"
    labeled_mesh: LabeledMesh
    try:
        labeled_mesh = segment(mesh)
    except Exception as exc:
        labeled_mesh = mesh  # degrade gracefully
        warnings.append(f"{stage}: {exc} — proceeding without labels")
    _tock(t1, stage)

    # ── Stage 3: Skeleton ─────────────────────────────────────────────
    t2 = time.perf_counter()
    stage = "predict_skeleton"
    skeleton: Skeleton
    try:
        skeleton = predict_skeleton(labeled_mesh)
    except Exception as exc:
        return PipelineResult(
            exit_code=2,
            output_path="",
            errors=[f"{stage}: {exc}"],
            warnings=warnings,
            stage_times={**stage_times, stage: time.perf_counter() - t2},
        )
    _tock(t2, stage)

    # ── Stage 4: Weights ──────────────────────────────────────────────
    t3 = time.perf_counter()
    stage = "compute_weights"
    weights: SkinWeights
    try:
        weights = compute_weights(labeled_mesh, skeleton)
    except Exception as exc:
        return PipelineResult(
            exit_code=2,
            output_path="",
            errors=[f"{stage}: {exc}"],
            warnings=warnings,
            stage_times={**stage_times, stage: time.perf_counter() - t3},
        )
    _tock(t3, stage)

    # ── Stage 5: Expressions ─────────────────────────────────────────
    t4 = time.perf_counter()
    stage = "create_expressions"
    expressions: list[Expression]
    try:
        expressions = create_expressions(labeled_mesh, skeleton)
    except Exception as exc:
        expressions = []  # degrade gracefully
        warnings.append(f"{stage}: {exc} — no expressions will be exported")
    _tock(t4, stage)

    # Convert list → dict keyed by expression name (required by exporter)
    expressions_dict: dict[str, Expression] = {e.name: e for e in expressions}

    # Assign morph_target_index sequentially
    for idx, expr in enumerate(expressions):
        expr.morph_target_index = idx

    # ── Stage 6: Spring Bones ─────────────────────────────────────────
    t5 = time.perf_counter()
    stage = "detect_spring_bones"
    spring_chains: list[SpringChain] = []
    colliders: list[Collider] = []
    try:
        spring_chains, colliders = detect_spring_bones(labeled_mesh, skeleton)
    except Exception as exc:
        warnings.append(f"{stage}: {exc} — spring bones skipped")
    _tock(t5, stage)

    # ── Stage 7: LookAt + FirstPerson ────────────────────────────────
    t6 = time.perf_counter()
    stage = "create_lookat"
    lookat: LookAt | None = None
    try:
        lookat = create_lookat(skeleton)
    except Exception as exc:
        warnings.append(f"{stage}: {exc} — lookAt skipped")
    _tock(t6, stage)

    t7 = time.perf_counter()
    stage = "create_firstperson"
    firstperson: FirstPerson | None = None
    try:
        firstperson = create_firstperson(skeleton, labeled_mesh)
    except Exception as exc:
        warnings.append(f"{stage}: {exc} — firstPerson skipped")
    _tock(t7, stage)

    # ── Stage 8: Export ───────────────────────────────────────────────
    t8 = time.perf_counter()
    stage = "export_vrm"
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_vrm(
            mesh=labeled_mesh,
            skeleton=skeleton,
            weights=weights,
            expressions=expressions_dict,
            spring_chains=spring_chains,
            colliders=colliders,
            lookat=lookat,
            firstperson=firstperson,
            meta=meta,
            output_path=output_path,
        )
    except Exception as exc:
        return PipelineResult(
            exit_code=2,
            output_path="",
            errors=[f"{stage}: {exc}"],
            warnings=warnings,
            stage_times={**stage_times, stage: time.perf_counter() - t8},
        )
    _tock(t8, stage)

    # ── QA: Geometry ───────────────────────────────────────────────────
    t_qa_geo = time.perf_counter()
    try:
        geo_errors = check_geometry(labeled_mesh)
        if geo_errors:
            warnings.extend(geo_errors)
    except Exception as exc:
        warnings.append(f"check_geometry: {exc}")
    stage_times["qa_geometry"] = time.perf_counter() - t_qa_geo

    # ── QA: Rig ────────────────────────────────────────────────────────
    t_qa_rig = time.perf_counter()
    try:
        rig_errors = check_rig(skeleton, weights, labeled_mesh)
        if rig_errors:
            warnings.extend(rig_errors)
    except Exception as exc:
        warnings.append(f"check_rig: {exc}")
    stage_times["qa_rig"] = time.perf_counter() - t_qa_rig

    # ── VRM schema validation ─────────────────────────────────────────
    t_qa_val = time.perf_counter()
    try:
        vrm_errors = validate_vrm(output_path)
        if vrm_errors:
            warnings.extend(vrm_errors)
    except Exception as exc:
        warnings.append(f"validate_vrm: {exc}")
    stage_times["validate_vrm"] = time.perf_counter() - t_qa_val

    return PipelineResult(
        exit_code=0,
        output_path=output_path,
        errors=errors,
        warnings=warnings,
        stage_times=stage_times,
    )
