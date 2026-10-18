# PCB Scan Stitcher — Design

Date: 2026-10-18

## Intent

Turn a grid of overlapping camera tiles from the CNC PCB scanner
(`pcb_scan_<id>/raw/tile_r{row}_c{col}.jpg` + `metadata.json`) into one
high-resolution mosaic. The two things that matter most:

1. **Trace correctness.** Copper traces, pads and silkscreen must line up
   across tile boundaries with no doubling, smearing or warping. A trace that
   crosses a seam must read as one continuous trace.
2. **Even coloring.** The mosaic must look uniformly lit: no visible tile
   checkerboard, no vignetting shadows at tile corners, consistent white
   balance across the board.

## What the data looks like (measured on `pcb_scan_1792339596`)

- 56 JPEG tiles, 9152×6944 px, Arducam 64 MP on a Raspberry Pi, Picamera2.
- 8 rows × 7 columns, serpentine path (rows alternate L2R / R2L).
- Stage steps 32.5 mm (x) and 22.143 mm (y). Measured pixel scale is
  ~80.5 px/mm on both axes, so the real field of view is ~114 × 86 mm and the
  overlap between neighbours is ~70 %, far more than the planned 35 %.
- Cross-axis offsets are small (~-25 px of y per x step, ~+5 px of x per
  y step): camera and stage axes are rotated by roughly half a degree.
- The bottom ~14 % of every frame shows static gantry hardware. It must be
  excluded from registration and from the mosaic.
- Focus, exposure and white balance were not locked, so tiles differ in gain
  and colour. Lens/illumination falloff is visible toward the frame corners.
- Tall components (capacitors, connectors) parallax between tiles; the board
  surface itself does not.

## Revision (same day): the geometry is not a translation

Measured on the real scan, dense NCC windows across a single overlap disagree
by over 100 px in a smooth pattern: the horizontal offset drifts linearly with
position and the vertical offset is antisymmetric top-to-bottom. That is radial
lens distortion (of order 1.5 %, over 100 px at the frame edges) combined with a
per-tile magnification change from unlocked autofocus (focus breathing). A
pure-translation mosaic therefore cannot keep traces continuous across seams,
whatever the blending does. The geometric model is now:

    mosaic = A_i · rectify(p)        rectify: shared smooth polynomial displacement
                                      (quadratic + cubic monomials in normalised
                                      coordinates: radial, tangential, decentering
                                      and tilted-board terms in one linear family)
    A_i: one 2×3 affine per tile      (stage translation + axis rotation + breathing)

solved jointly by bundle adjustment over window matches. Tiles are rendered
through the inverse warp (bicubic by default). The "no resampling" anti-goal is
withdrawn: resampling costs a little sharpness, misalignment breaks traces.

A second lesson from the real data: tall components parallax by tens to
hundreds of pixels between neighbouring tiles, and the cutting mat's grid lines
create ambiguous matches. Matches are therefore vetted three ways before the
bundle sees them: NCC peak sharpness along both axes, prominence over every
other candidate in the search area, and a per-pair RANSAC affine consensus
that keeps only board-plane correspondences.

## Goals

- One command: `pcbstitch <scan_dir>` → pyramidal tiled BigTIFF mosaic,
  a preview JPEG, a diagnostics report, and optionally a Deep Zoom pyramid.
- Shared lens model plus one affine per tile, fitted by bundle adjustment to
  dense, vetted window matches; sub-pixel consistency over the whole grid.
- Data-driven illumination model (flat-field + per-tile gain and gradient per
  colour channel), estimated from the overlaps themselves.
- Seam placement where each pixel comes from the tile in which it is most
  central (least vignetted, least distorted, least parallax); multi-band
  blending so colour blends over a wide band but detail never doubles.
- Memory-bounded compositing in blocks, so a true gigapixel scan still fits.
- Diagnostics: per-pair alignment quality, residuals, fallbacks, measured
  scale, and an annotated overview image.

## Non-goals

- Perspective (board tilt) or non-radial lens terms; the per-pair residuals in
  the report show whether they matter.
- Focus stacking, HDR, or de-parallaxing tall components.
- A GUI.

## Anti-goals (what would make this a failure)

- Warping that is not driven by measured correspondences (no "stretch to fit").
- Ghosting from feathered blends of slightly misaligned content.
- An alignment that "succeeds" by locking onto the static gantry bar.
- Silent fallbacks: every pair that could not be verified is reported.

## Pipeline

1. **Load session** from `metadata.json` waypoints (row, col, x_mm, y_mm) and
   `raw/tile_r*_c*.jpg`. Filenames alone are accepted when metadata is
   missing; grid indices then stand in for stage coordinates.
2. **Static-region detection.** Per-pixel standard deviation across all tiles
   at 1/16 scale; rows/columns that are mostly static are cropped away (plus a
   safety margin). The crop is applied to every tile for all later stages and
   can be overridden on the command line.
3. **Coarse registration** at 1/8 scale: phase correlation of whole cropped
   frames for every right- and down-neighbour pair.
4. **Stage model.** Robust affine fit from stage mm → pixels using the coarse
   offsets. Gives px/mm, axis rotation, and a prediction for every pair.
5. **Window matching** at full resolution: up to 60 windows of 256 px per
   pair, each warped through the current geometry onto the neighbour's pixel
   grid and located by NCC within a search radius, parabolic sub-pixel peak.
   Rejected: low score, peak on the search boundary, peak not sharp along
   both axes (aperture problem on lines), peak not prominent over other
   candidates (repetitive texture). Per pair, RANSAC keeps the largest
   affine-consistent set (drops parallax on tall components).
6. **Bundle adjustment.** Least squares over all matches for the shared
   rectification polynomial and per-tile affines, tile 0 fixed as gauge, weak
   priors to the stage model (plain least squares first, then soft-L1 with
   outlier rejection). Three passes: from the stage prior with a wide search
   radius; from the fitted geometry with a narrow radius; finally half-size
   windows with a tight per-pair consensus. The camera sits ~8 cm above the
   board, so surface relief alone parallaxes by about a pixel between
   neighbours: ~1 px rms is the physical floor, and only board-plane matches
   must survive to the last solve.
7. **Illumination model** (per channel, log domain, at 1/16 scale): for
   corresponding samples in each overlap,
   `log I_i − log I_j = F(u_i) − F(u_j) + (a_i − a_j) + gradient terms`,
   with `F` a smooth flat-field on a coarse bilinear grid (Laplacian
   regularised), `a_i` a per-tile log gain and a ridge-regularised per-tile
   linear gradient. Solved with IRLS (Cauchy weights) so speculars and
   parallax do not bias it. The scene term cancels, so no assumption about
   board content is needed.
8. **Seams.** Per output block, the inverse warp of every candidate tile is
   evaluated; each pixel goes to the tile in which it is most central
   (normalised distance from the tile centre in tile coordinates).
9. **Warp and blend** per output block with padding: tiles are resampled
   through the inverse warp (bicubic), illumination-corrected at their tile
   coordinates, then multi-band blended with the seam masks (Laplacian
   pyramids of the crops, Gaussian pyramids of the masks, per-level
   normalisation). Padding is at least the pyramid support.
10. **Output.** Mosaic assembled in a uint8 array (RAM, or a memmap on
    request), written as tiled pyramidal BigTIFF (JPEG q92 by default,
    lossless deflate on request), a ≤4096 px preview JPEG, optional DZI.
11. **Report.** `report.json` (scale, stage model, crop, per-pair offsets,
    scores, spreads, residuals, fallbacks, illumination gains) and
    `overview.png` (1/16 mosaic with tile boxes and pair links coloured by
    quality).

## Decisions made here (builder's calls)

- Python 3.11, numpy, scipy, OpenCV (headless) for image ops and NCC,
  pyvips for decoding (shrink-on-load) and for writing pyramids. `uv` manages
  the environment.
- Bicubic resampling through the fitted warp (linear and Lanczos selectable).
- Seam = nearest-centre Voronoi rather than minimum-error seams: with ~70 %
  overlap the most-central choice minimises vignetting, distortion residual
  and parallax, and the multi-band blend hides the rest.
- Default output block 2048 px with 384 px padding and 6 pyramid levels;
  threads share an LRU cache of decoded tiles.
- Flat-field normalised so its 99th percentile is 1 (corners are brightened
  up to the centre level, nothing is darkened); global exposure multiplier
  exposed as a flag.

## Open for the builder

- Exact thresholds (static std, NCC acceptance, outlier cutoffs) are
  configuration with sensible defaults; all appear in the report.
- Whether to also emit a lossless mosaic by default (disk is tight on this
  machine, so no).

## Verification

Synthetic end-to-end tests: generate a scene with thin traces, pads and text
blobs, cut it into a serpentine grid with jitter, backlash, vignetting,
per-tile gains, noise, a static bar, and optionally lens distortion and
focus breathing; stitch; assert sub-pixel match residuals, recovered
distortion and magnifications, and block-wise agreement with the ground
truth (traces reproduced, no ghosting at seams, brightness uniform).
Unit tests per module. Target ≥80 % coverage.
