"""VTuber Rigger CLI — command-line interface."""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from typing import Optional


def main() -> int:
    """Main CLI entry point.

    Commands:
        rig        — Path A: 3D model → VRM
        reconstruct — Path B: screenshots → VRM (Phase 4, not yet)
        verify     — validate a VRM file
        preview    — generate thumbnail
        info       — print VRM metadata
    """
    parser = argparse.ArgumentParser(
        prog="vtuber-rigger",
        description="AI auto-rigger for VTuber avatars",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # rig
    p_rig = sub.add_parser("rig", help="Path A: 3D model → VRM")
    p_rig.add_argument("input", help="Input model file (.glb, .gltf, .vrm, .fbx, .obj)")
    p_rig.add_argument("-o", "--output", required=True, help="Output .vrm file path")
    p_rig.add_argument("--quality", choices=["low", "medium", "high"], default="high")
    p_rig.add_argument("--name", default="VTuber Avatar", help="Avatar name")

    # reconstruct
    p_rec = sub.add_parser("reconstruct", help="Path B: screenshots → VRM")
    p_rec.add_argument("--front", help="Front view screenshot")
    p_rec.add_argument("--side", help="Side view screenshot")
    p_rec.add_argument("--back", help="Back view screenshot")
    p_rec.add_argument("--quarter", help="3/4 view screenshot")
    p_rec.add_argument("-o", "--output", required=True, help="Output .vrm file path")
    p_rec.add_argument("--name", default="VTuber Avatar", help="Avatar name")

    # verify
    p_ver = sub.add_parser("verify", help="Validate a VRM file")
    p_ver.add_argument("input", help="VRM file to validate")

    # preview
    p_pre = sub.add_parser("preview", help="Generate thumbnail from VRM")
    p_pre.add_argument("input", help="VRM file")
    p_pre.add_argument("-o", "--output", default="thumbnail.png", help="Output PNG path")
    p_pre.add_argument("--size", type=int, default=512, help="Thumbnail size (pixels)")

    # info
    p_info = sub.add_parser("info", help="Print VRM metadata")
    p_info.add_argument("input", help="VRM file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "rig":
            return _cmd_rig(args)
        elif args.command == "reconstruct":
            return _cmd_reconstruct(args)
        elif args.command == "verify":
            return _cmd_verify(args)
        elif args.command == "preview":
            return _cmd_preview(args)
        elif args.command == "info":
            return _cmd_info(args)
        else:
            parser.print_help()
            return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except NotImplementedError as e:
        print(f"Not implemented: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        return 2


def _cmd_rig(args) -> int:
    """Path A: 3D model → VRM"""
    from vtuber_rigger.pipeline.a_pipeline import run_pipeline_a
    from vtuber_rigger.interfaces import VRMMeta

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    meta = VRMMeta(name=args.name, authors=["VTuber Rigger"])
    result = run_pipeline_a(args.input, args.output, meta)

    if result.exit_code == 0:
        print(f"✅ VRM generated: {result.output_path}")
        if result.warnings:
            print(f"⚠️  {len(result.warnings)} warnings")
            for w in result.warnings[:5]:
                print(f"   {w}")
        for stage, t in result.stage_times.items():
            print(f"   {stage}: {t:.2f}s")
        return 0
    else:
        print(f"❌ Pipeline failed:", file=sys.stderr)
        for err in result.errors:
            print(f"   {err}", file=sys.stderr)
        return 3


def _cmd_reconstruct(args) -> int:
    """Path B: screenshots → VRM"""
    from vtuber_rigger.pipeline.b_pipeline import run_pipeline_b
    from vtuber_rigger.interfaces import VRMMeta

    screenshots = {}
    for name in ["front", "side", "back", "quarter"]:
        path = getattr(args, name, None)
        if path:
            if not os.path.exists(path):
                print(f"Error: {name} view not found: {path}", file=sys.stderr)
                return 1
            screenshots[name] = path

    if not screenshots:
        print("Error: at least one view required (--front, --side, --back, --quarter)", file=sys.stderr)
        return 1

    meta = VRMMeta(name=args.name, authors=["VTuber Rigger"])
    result = run_pipeline_b(screenshots, args.output, meta)

    if result.exit_code == 0:
        print(f"✅ VRM generated: {result.output_path}")
        return 0
    else:
        print(f"❌ Pipeline failed:", file=sys.stderr)
        for err in result.errors:
            print(f"   {err}", file=sys.stderr)
        return 3


def _cmd_verify(args) -> int:
    """Validate a VRM file."""
    from vtuber_rigger.vrm.exporter import validate_vrm

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1

    errors = validate_vrm(args.input)
    if not errors:
        print(f"✅ {args.input} is a valid VRM 1.0 file")
        return 0
    else:
        print(f"❌ {args.input} has {len(errors)} validation errors:")
        for err in errors:
            print(f"   {err}")
        return 3


def _cmd_preview(args) -> int:
    """Generate thumbnail from VRM."""
    from vtuber_rigger.qa.preview import generate_thumbnail

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        path = generate_thumbnail(args.input, args.output, args.size)
        print(f"✅ Thumbnail generated: {path}")
        return 0
    except Exception as e:
        print(f"❌ Thumbnail generation failed: {e}", file=sys.stderr)
        return 2


def _cmd_info(args) -> int:
    """Print VRM metadata."""
    from vtuber_rigger.qa.schema import parse_vrm

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        vrm = parse_vrm(args.input)
        meta = vrm.get("extensions", {}).get("VRMC_vrm", {}).get("meta", {})
        humanoid = vrm.get("extensions", {}).get("VRMC_vrm", {}).get("humanoid", {})
        bones = humanoid.get("humanBones", {})

        print(f"File: {args.input}")
        print(f"Size: {os.path.getsize(args.input) / 1e6:.2f} MB")
        print(f"Name: {meta.get('name', 'N/A')}")
        print(f"Version: {meta.get('version', 'N/A')}")
        print(f"Authors: {', '.join(meta.get('authors', []))}")
        print(f"License: {meta.get('licenseUrl', 'N/A')}")
        print(f"Humanoid bones: {len(bones)}")
        return 0
    except Exception as e:
        print(f"Error reading VRM: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())