"""End-to-end stitching pipeline."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import __version__
from .bundle import BundleResult, solve_bundle
from .camera import MosaicModel, identity_camera
from .compose import INTERPOLATION, ComposeConfig, Composer
from .errors import ConfigError, RegistrationError
from .geometry import Rect
from .illumination import IlluminationModel, fit_illumination
from .matching import Match, PairMapper, match_pair_windows, ransac_affine, translation_mapper
from .output import downscale, write_dzi, write_png, write_preview, write_tiff
from .registration import CoarseOffset, auto_coarse_scale, phase_correlation
from .report import draw_overview, pair_stats, summarise_pairs, write_report
from .scan import Pair, ScanSession, load_session
from .stagemodel import StageModel, fit_stage_model
from .static import detect_static_crop
from .tiles import TileSource

log = logging.getLogger(__name__)

OVERVIEW_SCALE = 16


@dataclass(frozen=True)
class StitchConfig:
    out_dir: Path | None = None
    crop: Rect | None = None
    coarse_scale: int | None = None
    window: int = 256
    radius: int = 220
    refine_radius: int = 24
    max_windows: int = 60
    min_score: float = 0.5
    min_distinct: float = 0.03
    ransac_px: float = 12.0
    refine_ransac_px: float = 2.5
    final_ransac_px: float = 2.0
    passes: int = 3
    distortion: bool = True
    stage_outlier_px: float = 12.0
    illumination: bool = True
    illum_scale: int = 16
    illum_grid: tuple[int, int] = (24, 18)
    chroma_ridge: float = 1.0
    block: int = 2048
    pad: int = 384
    levels: int = 6
    workers: int = 4
    exposure: float = 1.0
    interpolation: str = "cubic"
    quality: int = 92
    lossless: bool = False
    dzi: bool = False
    preview_dim: int = 4096
    cache_bytes: int = 4 << 30
    memmap: bool = False
    dry_run: bool = False
    max_rms_px: float = 8.0
    max_mosaic_bytes: int = 12 << 30


@dataclass
class StitchResult:
    out_dir: Path
    width: int
    height: int
    report: dict
    report_path: Path
    model: MosaicModel
    mosaic_path: Path | None = None
    preview_path: Path | None = None
    overview_path: Path | None = None
    dzi_path: Path | None = None
    mosaic: np.ndarray | None = field(default=None, repr=False)


class _Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round(now - self.t0, 2)
        log.info("[%6.1fs] %s done", now - self.t0, name)


def _pair_order(session: ScanSession, pairs: tuple[Pair, ...]) -> list[int]:
    """Pairs sorted by grid row then column so the tile cache sees locality."""
    t = session.tiles
    return sorted(range(len(pairs)), key=lambda k: (min(t[pairs[k].i].row, t[pairs[k].j].row), t[pairs[k].i].col))


def coarse_offsets(source: TileSource, session: ScanSession, scale: int, workers: int) -> list[CoarseOffset]:
    pairs = session.neighbor_pairs()

    def one(p: Pair) -> CoarseOffset:
        off = phase_correlation(source.gray(p.i, scale), source.gray(p.j, scale))
        return CoarseOffset(p.i, p.j, p.axis, off.dx * scale, off.dy * scale, off.score)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, pairs))


def fit_stage(session: ScanSession, coarse: list[CoarseOffset], outlier_px: float) -> tuple[StageModel, np.ndarray]:
    stage = session.stage_coords()
    ds = np.array([stage[m.j] - stage[m.i] for m in coarse])
    d = np.array([[m.dx, m.dy] for m in coarse])
    w = np.array([max(m.score, 1e-3) ** 2 for m in coarse])
    return fit_stage_model(ds, d, w, outlier_px=outlier_px)


MIN_PAIR_MATCHES = 4


def collect_matches(
    source: TileSource, session: ScanSession, mapper_for, cfg: StitchConfig, radius: int, ransac_px: float,
    previous: list[Match] | None = None,
) -> list[Match]:
    """Window matches for every neighbour pair; ``mapper_for(pair)`` gives the prior geometry.

    Each pair's matches are reduced to the largest affine-consistent subset so that
    parallax on tall components and mismatches on repetitive texture never reach
    the bundle adjustment. When a pair ends up with fewer than ``MIN_PAIR_MATCHES``
    matches and ``previous`` holds more for it, the previous matches are kept.
    """
    pairs = session.neighbor_pairs()
    order = _pair_order(session, pairs)
    prev_by_pair: dict[tuple[int, int], list[Match]] = {}
    for m in previous or []:
        prev_by_pair.setdefault((m.i, m.j), []).append(m)

    def one(k: int) -> list[Match]:
        p = pairs[k]
        raw = match_pair_windows(
            source.load(p.i), source.load(p.j), p.i, p.j, mapper_for(p),
            window=cfg.window, radius=radius, max_windows=cfg.max_windows,
            min_score=cfg.min_score, min_distinct=cfg.min_distinct,
        )
        kept = ransac_affine(raw, ransac_px)
        if raw and len(kept) < len(raw):
            log.debug("pair %d-%d: %d/%d matches affine-consistent", p.i, p.j, len(kept), len(raw))
        return kept

    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        results = list(pool.map(one, order))
    by_index = dict(zip(order, results))
    matches: list[Match] = []
    for k, p in enumerate(pairs):
        found = by_index[k]
        kept_prev = prev_by_pair.get((p.i, p.j), [])
        if len(found) < MIN_PAIR_MATCHES and len(kept_prev) > len(found):
            log.debug("pair %d-%d: %d new matches; keeping %d from the previous pass", p.i, p.j, len(found), len(kept_prev))
            found = kept_prev
        if not found:
            log.warning("pair %d-%d: no usable window match; relying on priors", p.i, p.j)
        matches.extend(found)
    return matches


def register(source: TileSource, session: ScanSession, cfg: StitchConfig, timer: _Timer) -> tuple[StageModel, BundleResult, list[Match]]:
    """Coarse offsets → stage model → two rounds of window matching + bundle adjustment."""
    th, tw = source.shape()
    coarse_scale = cfg.coarse_scale or auto_coarse_scale(tw, th)
    coarse = coarse_offsets(source, session, coarse_scale, cfg.workers)
    source.drop_scale(coarse_scale)
    timer.mark(f"coarse registration (1/{coarse_scale})")
    if not coarse:
        raise RegistrationError("the scan has a single tile or no neighbouring tiles")
    stage_model, _ = fit_stage(session, coarse, cfg.stage_outlier_px)
    if stage_model.n_inliers < 1 or not np.isfinite(stage_model.px_per_unit):
        raise RegistrationError("could not fit a stage model from coarse offsets")
    timer.mark("stage model")

    stage = session.stage_coords()
    camera = identity_camera(tw, th, source.full_width, source.full_height, source.crop)
    initial = MosaicModel.from_translations(camera, stage_model.predict(stage - stage[0]), tw, th)

    def stage_mapper(p: Pair) -> PairMapper:
        d = stage_model.predict(stage[p.j] - stage[p.i])
        return translation_mapper(float(d[0]), float(d[1]))

    def model_mapper(model: MosaicModel):
        return lambda p: PairMapper(model.pair_map(p.i, p.j), model.pair_map(p.j, p.i))

    matches = collect_matches(source, session, stage_mapper, cfg, cfg.radius, cfg.ransac_px)
    if not matches:
        raise RegistrationError("no window matches found; check --radius, --min-score and the static crop")
    timer.mark(f"window matching pass 1 ({len(matches)} matches)")
    result = solve_bundle(matches, initial, fit_distortion=cfg.distortion)
    timer.mark("bundle adjustment pass 1")

    # Later passes: templates warped through the fitted geometry, narrower search,
    # tighter consensus and (last pass) smaller windows so parallax on components
    # is excluded and only board-plane correspondences remain.
    for k in range(2, max(cfg.passes, 1) + 1):
        last = k == cfg.passes
        if last:
            ransac, radius, outlier = cfg.final_ransac_px, max(cfg.refine_radius // 2, 8), max(2.0, cfg.final_ransac_px)
            pass_cfg = replace(cfg, window=max(cfg.window // 2, 48), max_windows=cfg.max_windows * 3)
        else:
            ransac, radius, outlier = max(cfg.refine_ransac_px, 1.5 * result.rms), cfg.refine_radius, 4.0
            pass_cfg = cfg
        new_matches = collect_matches(source, session, model_mapper(result.model), pass_cfg, radius, ransac, previous=matches)
        if not new_matches:
            log.warning("pass %d found no matches; keeping the previous geometry", k)
            break
        matches = new_matches
        timer.mark(f"window matching pass {k} ({len(matches)} matches)")
        result = solve_bundle(matches, result.model, fit_distortion=cfg.distortion, outlier_px=outlier)
        timer.mark(f"bundle adjustment pass {k}")
    final = result
    return stage_model, final, matches


SCRATCH_NAME = "mosaic_scratch.npy"


def _allocate_mosaic(width: int, height: int, out_dir: Path, memmap: bool) -> np.ndarray:
    if memmap:
        path = out_dir / SCRATCH_NAME
        return np.lib.format.open_memmap(str(path), mode="w+", dtype=np.uint8, shape=(height, width, 3))
    return np.zeros((height, width, 3), np.uint8)


def _validate_config(cfg: StitchConfig) -> None:
    if cfg.interpolation not in INTERPOLATION:
        raise ConfigError(f"interpolation must be one of {sorted(INTERPOLATION)}, got {cfg.interpolation!r}")
    if cfg.block < 64 or cfg.levels < 1 or cfg.workers < 1 or cfg.window < 16 or cfg.passes < 1:
        raise ConfigError("block, levels, workers, window and passes must be positive (block >= 64, window >= 16)")
    if not 1 <= cfg.quality <= 100:
        raise ConfigError("quality must be between 1 and 100")


def _check_registration(bundle: BundleResult, n_tiles: int, width: int, height: int, cfg: StitchConfig) -> None:
    """Refuse to render a mosaic from a registration that is clearly broken."""
    n_inliers = int(bundle.inliers.sum())
    if not np.isfinite(bundle.rms) or bundle.rms > cfg.max_rms_px:
        raise RegistrationError(f"alignment residual {bundle.rms:.1f} px exceeds --max-rms-px {cfg.max_rms_px}; "
                                "check the overview/report or relax the limit deliberately")
    if n_tiles > 1 and n_inliers < n_tiles:
        raise RegistrationError(f"only {n_inliers} usable matches for {n_tiles} tiles; the geometry is unconstrained")
    nbytes = width * height * 3
    if nbytes > cfg.max_mosaic_bytes and not cfg.memmap:
        raise ConfigError(f"mosaic would need {nbytes / 2**30:.1f} GiB in RAM; use --memmap or raise --max-mosaic-gb")


def stitch(scan_dir: str | Path, cfg: StitchConfig = StitchConfig()) -> StitchResult:
    _validate_config(cfg)
    timer = _Timer()
    scan_dir = Path(scan_dir)
    session = load_session(scan_dir)
    out_dir = cfg.out_dir or (scan_dir / "stitched")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("session %s: %d tiles, %d rows × %d cols", scan_dir.name, len(session.tiles), session.rows, session.cols)

    probe = TileSource(session, cache_bytes=cfg.cache_bytes)
    if cfg.crop is not None:
        crop = cfg.crop
        static_info: dict = {"mode": "manual"}
    else:
        det = detect_static_crop(probe)
        crop = det.crop
        static_info = {"mode": "auto", "threshold": det.threshold, "scale": det.scale}
    source = TileSource(session, crop=crop, cache_bytes=cfg.cache_bytes)
    th, tw = source.shape()
    log.info("tile crop %s → %dx%d usable pixels", crop.as_tuple(), tw, th)
    timer.mark("static detection")

    stage_model, bundle, matches = register(source, session, cfg, timer)
    model = bundle.model.normalized()
    ext = model.extent()
    width, height = ext.x1, ext.y1
    log.info("mosaic size %dx%d (%.2f gigapixels)", width, height, width * height / 1e9)
    _check_registration(bundle, len(session.tiles), width, height, cfg)

    illum: IlluminationModel | None = None
    pairs = np.array([[p.i, p.j] for p in session.neighbor_pairs()]).reshape(-1, 2)
    if cfg.illumination:
        illum = fit_illumination(source, model, pairs, scale=cfg.illum_scale, grid=cfg.illum_grid,
                                 chroma_ridge=cfg.chroma_ridge)
        source.drop_scale(cfg.illum_scale)
        timer.mark("illumination model")

    stats = pair_stats(matches, bundle.residuals, bundle.inliers)
    report = {
        "pcbstitch_version": __version__,
        "scan_dir": str(scan_dir),
        "n_tiles": len(session.tiles),
        "rows": session.rows,
        "cols": session.cols,
        "tile_size_px": [source.full_width, source.full_height],
        "crop": crop.as_tuple(),
        "static_detection": static_info,
        "stage_model": stage_model.to_dict(),
        "stage_units": "mm" if session.has_stage_coords() else "grid",
        "geometry": model.to_dict(),
        "mosaic_size_px": [width, height],
        "gigapixels": width * height / 1e9,
        "tiles": [
            {"index": t.index, "row": t.row, "col": t.col, "file": t.path.name,
             "bbox": model.tile_bbox(t.index).as_tuple(),
             "scale": float(model.scales()[t.index]),
             "rotation_deg": float(model.rotations_deg()[t.index])}
            for t in session.tiles
        ],
        "matches_summary": summarise_pairs(stats, bundle),
        "pairs": stats,
        "illumination": illum.to_dict() if illum is not None else None,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
    }

    if cfg.dry_run:
        report["timings_s"] = timer.marks
        report_path = write_report(report, out_dir / "report.json")
        return StitchResult(out_dir, width, height, report, report_path, model)

    mosaic = _allocate_mosaic(width, height, out_dir, cfg.memmap)
    composer = Composer(source, model, illum,
                        ComposeConfig(cfg.block, cfg.pad, cfg.levels, cfg.workers, cfg.exposure, cfg.interpolation))

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            log.info("compositing %d/%d blocks", done, total)

    composer.run(mosaic, progress)
    timer.mark("compositing")

    mosaic_path = write_tiff(mosaic, out_dir / "mosaic.tif", cfg.quality, cfg.lossless)
    preview_path = write_preview(mosaic, out_dir / "preview.jpg", cfg.preview_dim)
    dzi_path = write_dzi(mosaic, out_dir / "mosaic_dzi") if cfg.dzi else None
    overview_scale = OVERVIEW_SCALE if max(width, height) >= OVERVIEW_SCALE * 64 else 1
    small = downscale(mosaic, overview_scale)
    overview_path = write_png(draw_overview(small, overview_scale, model, stats), out_dir / "overview.png")
    timer.mark("output")
    report["timings_s"] = timer.marks
    report["outputs"] = {"mosaic": mosaic_path.name, "preview": preview_path.name, "overview": overview_path.name,
                         "dzi": dzi_path.name if dzi_path else None}
    report_path = write_report(report, out_dir / "report.json")
    if cfg.memmap:
        del mosaic  # release the mapping before removing the scratch file
        (out_dir / SCRATCH_NAME).unlink(missing_ok=True)
        mosaic = None
    return StitchResult(out_dir, width, height, report, report_path, model, mosaic_path, preview_path,
                        overview_path, dzi_path, mosaic)
