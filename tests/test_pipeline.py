import json

import numpy as np
import pyvips
import pytest

from pcbstitch.cli import main
from pcbstitch.pipeline import StitchConfig, stitch

from .synth import COPPER_RGB, default_scan

SMALL = dict(window=48, radius=16, refine_radius=6, max_windows=24, illum_scale=2, illum_grid=(12, 9),
             block=256, pad=64, levels=4, workers=2, quality=95, preview_dim=512)


def _stitch(tmp_path, scan_kw=None, **overrides):
    scan, root = default_scan(tmp_path, **(scan_kw or {}))
    cfg = StitchConfig(**{**SMALL, **overrides})
    return scan, root, stitch(root, cfg)


def _check_against_truth(scan, result, max_trace_err=15.0, max_shift=8, min_ncc=0.9, max_cv=0.04):
    """Local correctness: every block of the mosaic must match the ground truth up to a
    small rigid shift (a smooth drift of a fraction of a percent is a gauge freedom
    between the shared rectification and the per-tile affines; it grows with the
    distance from tile 0), with traces reproduced and no ghosting at seams."""
    import cv2

    mosaic = result.mosaic.astype(np.float32)
    my, mx = np.mgrid[0 : result.height, 0 : result.width].astype(np.float64)
    ref = scan.scene_at(result.model, mx, my).astype(np.float32)
    covered = mosaic.sum(axis=2) > 0
    k = np.median(ref[covered]) / max(np.median(mosaic[covered]), 1e-6)
    got = mosaic * k
    gray_got = cv2.GaussianBlur(cv2.cvtColor(got, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    gray_ref = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    b, m = 64, max_shift
    trace_errs, nccs, ratios = [], [], []
    for by in range(m, result.height - b - m, b):
        for bx in range(m, result.width - b - m, b):
            if covered[by - m : by + b + m, bx - m : bx + b + m].mean() < 0.999:
                continue
            res = cv2.matchTemplate(gray_ref[by - m : by + b + m, bx - m : bx + b + m], gray_got[by : by + b, bx : bx + b],
                                    cv2.TM_CCOEFF_NORMED)
            py, px = np.unravel_index(int(np.argmax(res)), res.shape)
            if gray_ref[by : by + b, bx : bx + b].std() > 8:  # featureless noise does not survive JPEG
                nccs.append(float(res[py, px]))
            sy, sx = by - m + py, bx - m + px
            r = ref[sy : sy + b, sx : sx + b]
            g = got[by : by + b, bx : bx + b]
            copper = np.all(np.abs(r - np.array(COPPER_RGB, np.float32)) < 40, axis=2)
            if copper.sum() > 20:
                trace_errs.append(np.abs(g - r)[copper].mean())
            ratios.append(g.mean() / r.mean())
    assert len(nccs) > 20 and len(trace_errs) > 5
    assert min(nccs) > min_ncc, f"worst block NCC {min(nccs):.3f}"
    assert np.mean(trace_errs) < max_trace_err, f"trace error {np.mean(trace_errs):.1f}"
    ratios = np.array(ratios)
    assert ratios.std() / ratios.mean() < max_cv, f"brightness CV {ratios.std() / ratios.mean():.3f}"


def test_end_to_end_translation_only(tmp_path):
    scan, root, result = _stitch(tmp_path)
    report = json.loads(result.report_path.read_text())
    assert report["matches_summary"]["residual_rms_px"] < 0.5
    assert report["matches_summary"]["n_weak_pairs"] == 0
    assert report["geometry"]["max_distortion_px"] < 3.0
    assert np.abs(np.array(report["geometry"]["tile_scales"]) - 1.0).max() < 0.005
    crop = report["crop"]
    assert crop[3] <= 240 - scan.static_rows and crop[:3] == [0, 0, 320]
    assert result.mosaic.shape == (result.height, result.width, 3)
    _check_against_truth(scan, result)
    tif = pyvips.Image.new_from_file(str(result.mosaic_path))
    assert (tif.width, tif.height, tif.bands) == (result.width, result.height, 3)
    assert result.preview_path.exists() and result.overview_path.exists()


def test_end_to_end_distortion_and_focus_breathing(tmp_path):
    scan_kw = dict(k1=0.025, scale_jitter=0.015, tilt=(0.01, -0.006), seed=3)
    scan, root, result = _stitch(tmp_path, scan_kw=scan_kw)
    geo = result.report["geometry"]
    assert result.report["matches_summary"]["residual_rms_px"] < 0.8
    assert abs(geo["max_distortion_px"] - scan.camera().max_displacement(320, 176)) < 3.0
    truth = scan.truth_model()
    # absolute scales are only loosely identifiable on 320 px tiles (gauge with the
    # optical centre); tile-to-tile consistency is what the residual rms checks
    assert np.abs(np.array(geo["tile_scales"]) - truth.scales()).max() < 0.015
    _check_against_truth(scan, result, max_trace_err=16.0)


def test_no_illumination_and_no_distortion_options(tmp_path):
    scan, root, result = _stitch(tmp_path, illumination=False, distortion=False, out_dir=tmp_path / "out2")
    assert result.report["illumination"] is None
    assert not np.any(result.report["geometry"]["camera"]["coefficients"])
    assert result.out_dir == tmp_path / "out2"
    assert result.report["matches_summary"]["residual_rms_px"] < 0.6


def test_dry_run_memmap_dzi_lossless(tmp_path):
    scan, root, result = _stitch(tmp_path, dry_run=True)
    assert result.mosaic is None and result.mosaic_path is None and result.report_path.exists()
    scan, root, result = _stitch(tmp_path, memmap=True, dzi=True, lossless=True, interpolation="lanczos",
                                 out_dir=tmp_path / "mm")
    assert result.mosaic is None and not (tmp_path / "mm" / "mosaic_scratch.npy").exists()
    assert result.mosaic_path.exists() and result.dzi_path.exists() and (tmp_path / "mm" / "mosaic_dzi_files").is_dir()


def test_manual_crop_and_missing_metadata(tmp_path):
    from pcbstitch.geometry import Rect

    scan, root = default_scan(tmp_path)
    (root / "metadata.json").unlink()
    result = stitch(root, StitchConfig(**SMALL, crop=Rect(0, 0, 320, 200), dry_run=True))
    assert result.report["stage_units"] == "grid"
    assert result.report["crop"] == (0, 0, 320, 200)
    assert result.report["static_detection"]["mode"] == "manual"


def test_registration_failure_is_reported(tmp_path):
    from pcbstitch.errors import ConfigError, RegistrationError

    scan, root = default_scan(tmp_path, rows=1, cols=1)
    with pytest.raises(RegistrationError):
        stitch(root, StitchConfig(**SMALL, dry_run=True))
    scan, root = default_scan(tmp_path / "b")
    with pytest.raises(RegistrationError):
        stitch(root, StitchConfig(**SMALL, max_rms_px=0.001))
    with pytest.raises(ConfigError):
        stitch(root, StitchConfig(**SMALL, max_mosaic_bytes=1000))
    with pytest.raises(ConfigError):
        stitch(root, StitchConfig(**SMALL, interpolation="nearest"))
    with pytest.raises(ConfigError):
        stitch(root, StitchConfig(**SMALL, quality=0))


def test_cli_runs_and_reports(tmp_path, capsys):
    scan, root = default_scan(tmp_path)
    rc = main([str(root), "--window", "48", "--radius", "16", "--refine-radius", "6", "--illum-scale", "2",
               "--illum-grid", "12x9", "--block", "256", "--pad", "64", "--levels", "4", "--workers", "2",
               "--preview-dim", "512", "--quality", "95", "--max-windows", "24"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mosaic:" in out and "alignment:" in out and (root / "stitched" / "mosaic.tif").exists()
    assert main([str(tmp_path / "missing")]) == 2
    assert main([str(root), "--crop", "0,0,9999,10", "--dry-run"]) == 2
    with pytest.raises(SystemExit):
        main([str(root), "--crop", "1,2,3"])
    with pytest.raises(SystemExit):
        main([str(root), "--illum-grid", "3"])
    with pytest.raises(SystemExit):
        main([str(root), "--crop", "5,5,5,9"])
