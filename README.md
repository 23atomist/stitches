# pcbstitch

Stitches the overlapping camera tiles from the CNC PCB scanner into one
seamless, evenly lit, high-resolution mosaic.

```
pcb_scan_<id>/
  metadata.json          # scan plan: rows, cols, stage x_mm/y_mm per tile
  raw/tile_r{row}_c{col}.jpg
  stitched/              # output (created)
    mosaic.tif           # tiled, pyramidal BigTIFF (JPEG q92 by default)
    preview.jpg          # ≤ 4096 px preview
    overview.png         # 1/16 mosaic with tile boxes and pair quality
    report.json          # everything measured along the way
```

## Usage

```bash
uv sync                      # once; needs libvips installed (brew install vips)
uv run pcbstitch pcb_scan_1792339596
uv run pcbstitch pcb_scan_1792339596 --dry-run     # register + report only
uv run pcbstitch pcb_scan_1792339596 --dzi --lossless --out /path/to/out
uv run pcbstitch --help
```

Open `mosaic.tif` in QuPath, vips, ImageJ, GIMP or any BigTIFF-aware viewer.
`--dzi` also writes a Deep Zoom pyramid for OpenSeadragon.

## What it does

1. **Static-region detection.** Pixels that never change between tiles are
   scanner hardware in the field of view; those rows and columns are cropped
   from every tile (override with `--crop x0,y0,x1,y1`).
2. **Coarse registration** by phase correlation at 1/8 scale, then a robust
   affine fit from stage millimetres to pixels (reported as px/mm and axis
   rotation).
3. **Window matching.** Up to 60 windows of 256 px per neighbour pair are
   located in the other tile by normalised cross-correlation. Matches are
   rejected when weak, on the search boundary, not sharp along both axes
   (lines), or not prominent over other candidates (repetitive texture), and
   each pair keeps only its largest affine-consistent set (drops parallax on
   tall components).
4. **Bundle adjustment** of a shared camera rectification (a smooth polynomial
   covering radial and tangential lens distortion and board tilt) plus one
   affine per tile (stage move, axis rotation, focus-breathing magnification).
   Three passes: wide search from the stage prior; narrow search from the
   fitted geometry with templates warped through it; finally half-size windows
   with a tight per-pair consensus so that only board-plane correspondences
   remain (surface relief and components parallax between tiles because the
   camera is only ~8 cm above the board).
5. **Illumination model** fitted from the overlaps themselves: a smooth
   flat-field (vignetting, lighting falloff) plus per-tile, per-channel gain
   and gradient, solved robustly in the log domain. The scene content cancels,
   so no assumption about the board is needed.
6. **Compositing** in blocks: every pixel is taken from the tile in which it is
   most central, tiles are resampled through the inverse warp (bicubic),
   corrected, and multi-band blended so colour fades smoothly across seams
   while detail never doubles.

## Reading the report

- `stage_model.px_per_unit` — measured pixel scale; `rotation_deg` — camera vs
  stage axes.
- `geometry.max_distortion_px` — rectification at the frame corners;
  `geometry.tile_scales` — per-tile magnification (focus breathing and a
  tilted board both show up here).
- `matches_summary.residual_rms_px` — how well all tiles agree where they
  overlap (about 1 px is the floor set by surface relief parallax);
  `n_weak_pairs` — pairs with too few matches or >2 px rms.
- `pairs[]` — per-pair counts, mean NCC and residuals.
- `overview.png` — links between tiles coloured green/yellow/red by residual.

## Tuning

| Flag | When |
|------|------|
| `--radius` | stage prior is worse than ±220 px (check `stage_model.inlier_rms_px`) |
| `--window`, `--max-windows` | boards with little texture: smaller windows, more of them |
| `--min-score`, `--min-distinct` | lower for low-contrast boards, raise if red links appear |
| `--no-distortion` | lens already rectified |
| `--no-illumination`, `--exposure` | keep the camera's rendering, or brighten/darken globally |
| `--levels`, `--pad`, `--block`, `--workers`, `--cache-gb`, `--memmap` | memory / speed |
| `--interp linear|cubic|lanczos` | resampling filter |

## Scanner notes (from `pcb_scan_1792339596`)

- The real pixel scale is ~80 px/mm, so the true field of view is ~114 × 86 mm,
  not the 59.5 × 36 mm in the plan; neighbours overlap ~70 %. Fixing the plan's
  FOV would let a scan cover twice the area with the same number of tiles.
- Autofocus and exposure were not locked; the stitcher compensates for both,
  but locking them will improve sharpness and colour fidelity.
- The bottom ~14 % of every frame is gantry hardware and is cropped
  automatically.

## Development

```bash
uv run pytest --cov=pcbstitch     # synthetic end-to-end + unit tests
```

Design notes: `docs/superpowers/specs/2026-10-18-pcb-stitcher-design.md`.
