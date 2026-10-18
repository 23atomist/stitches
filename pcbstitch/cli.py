"""Command line entry point: ``pcbstitch SCAN_DIR [options]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .errors import StitchError
from .geometry import Rect
from .pipeline import StitchConfig, stitch


def _parse_crop(text: str) -> Rect:
    try:
        x0, y0, x1, y1 = (int(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1") from exc
    rect = Rect(x0, y0, x1, y1)
    if rect.is_empty:
        raise argparse.ArgumentTypeError("crop is empty")
    return rect


def _parse_grid(text: str) -> tuple[int, int]:
    try:
        a, b = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must look like 3x3") from exc
    if a < 1 or b < 1:
        raise argparse.ArgumentTypeError("grid must be at least 1x1")
    return a, b


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pcbstitch", description="Stitch PCB scanner tiles into one evenly lit mosaic.")
    p.add_argument("scan_dir", type=Path, help="scan directory with raw/tile_r*_c*.jpg and metadata.json")
    p.add_argument("-o", "--out", type=Path, default=None, help="output directory (default: SCAN_DIR/stitched)")
    p.add_argument("--crop", type=_parse_crop, default=None, help="usable tile region x0,y0,x1,y1 (skips static detection)")
    p.add_argument("--coarse-scale", type=int, default=None, help="power-of-two shrink for coarse registration (auto)")
    p.add_argument("--window", type=int, default=256, help="NCC window size in px for matching")
    p.add_argument("--radius", type=int, default=220, help="first-pass search radius in px around the stage prediction")
    p.add_argument("--refine-radius", type=int, default=24, help="second-pass search radius around the fitted geometry")
    p.add_argument("--max-windows", type=int, default=60, help="maximum matched windows per tile pair")
    p.add_argument("--min-score", type=float, default=0.5, help="minimum NCC for a window to count")
    p.add_argument("--min-distinct", type=float, default=0.03, help="minimum NCC peak prominence (rejects ambiguous matches)")
    p.add_argument("--ransac-px", type=float, default=12.0, help="first-pass per-pair affine consensus threshold")
    p.add_argument("--refine-ransac-px", type=float, default=2.5, help="second-pass per-pair affine consensus threshold")
    p.add_argument("--final-ransac-px", type=float, default=2.0, help="last-pass per-pair affine consensus threshold")
    p.add_argument("--passes", type=int, default=3, help="matching + bundle adjustment passes")
    p.add_argument("--no-distortion", action="store_true", help="do not fit lens distortion (affine per tile only)")
    p.add_argument("--interp", choices=["linear", "cubic", "lanczos"], default="cubic", help="tile resampling filter")
    p.add_argument("--no-illumination", action="store_true", help="skip flat-field / gain correction")
    p.add_argument("--illum-scale", type=int, default=16, help="shrink factor used to fit the illumination model")
    p.add_argument("--illum-grid", type=_parse_grid, default=(24, 18), help="flat-field grid nodes, e.g. 24x18")
    p.add_argument("--chroma-ridge", type=float, default=1.0, help="how strongly the flat-field is assumed achromatic (0 = per-channel free)")
    p.add_argument("--exposure", type=float, default=1.0, help="global brightness multiplier")
    p.add_argument("--levels", type=int, default=6, help="multi-band blending pyramid levels")
    p.add_argument("--block", type=int, default=2048, help="output block size in px")
    p.add_argument("--pad", type=int, default=384, help="block padding in px")
    p.add_argument("--workers", type=int, default=4, help="worker threads")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality for the TIFF pyramid")
    p.add_argument("--lossless", action="store_true", help="deflate-compressed TIFF instead of JPEG")
    p.add_argument("--dzi", action="store_true", help="also write a Deep Zoom pyramid")
    p.add_argument("--preview-dim", type=int, default=4096, help="longest side of preview.jpg")
    p.add_argument("--cache-gb", type=float, default=4.0, help="decoded tile cache size in GiB")
    p.add_argument("--memmap", action="store_true", help="back the mosaic with a file instead of RAM")
    p.add_argument("--max-rms-px", type=float, default=8.0, help="refuse to render when alignment residual exceeds this")
    p.add_argument("--max-mosaic-gb", type=float, default=12.0, help="largest mosaic to hold in RAM without --memmap")
    p.add_argument("--dry-run", action="store_true", help="register and report only; no mosaic")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def config_from_args(args: argparse.Namespace) -> StitchConfig:
    return StitchConfig(
        out_dir=args.out,
        crop=args.crop,
        coarse_scale=args.coarse_scale,
        window=args.window,
        radius=args.radius,
        refine_radius=args.refine_radius,
        max_windows=args.max_windows,
        min_score=args.min_score,
        min_distinct=args.min_distinct,
        ransac_px=args.ransac_px,
        refine_ransac_px=args.refine_ransac_px,
        final_ransac_px=args.final_ransac_px,
        passes=args.passes,
        distortion=not args.no_distortion,
        interpolation=args.interp,
        illumination=not args.no_illumination,
        illum_scale=args.illum_scale,
        illum_grid=args.illum_grid,
        chroma_ridge=args.chroma_ridge,
        block=args.block,
        pad=args.pad,
        levels=args.levels,
        workers=args.workers,
        exposure=args.exposure,
        quality=args.quality,
        lossless=args.lossless,
        dzi=args.dzi,
        preview_dim=args.preview_dim,
        cache_bytes=int(args.cache_gb * (1 << 30)),
        memmap=args.memmap,
        dry_run=args.dry_run,
        max_rms_px=args.max_rms_px,
        max_mosaic_bytes=int(args.max_mosaic_gb * (1 << 30)),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    try:
        result = stitch(args.scan_dir, config_from_args(args))
    except (StitchError, ValueError) as exc:
        logging.getLogger("pcbstitch").error("%s", exc)
        return 2
    summary = result.report["matches_summary"]
    geo = result.report["geometry"]
    print(f"mosaic: {result.width}x{result.height} px ({result.report['gigapixels']:.3f} GP)")
    print(f"scale: {result.report['stage_model']['px_per_unit']:.2f} px/{result.report['stage_units']}")
    print(f"geometry: lens/tilt rectification up to {geo['max_distortion_px']:.1f} px at the frame corners, "
          f"tile magnification {min(geo['tile_scales']):.4f}..{max(geo['tile_scales']):.4f}")
    print(f"alignment: {summary['n_inliers']}/{summary['n_matches']} matches used over {summary['n_pairs']} pairs, "
          f"residual rms {summary['residual_rms_px']:.2f} px, max {summary['residual_max_px']:.2f} px, "
          f"{summary['n_weak_pairs']} weak pairs")
    if result.mosaic_path is not None:
        print(f"wrote: {result.mosaic_path}")
        print(f"preview: {result.preview_path}")
    print(f"report: {result.report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
