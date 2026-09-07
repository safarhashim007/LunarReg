# LunarReg production validation — 7 September 2026

## Decision

Use the preserved, hash-verified V9 transform for pair_002. The new package and CLI are implemented and tested, with repeatable controlled validation. **This is a safe production entry point for the validated pair, not a certification of reliable automatic registration on arbitrary real pairs.** Fresh real-pair inference exposed a quality-gate false positive and is not promoted. V11 stays experimental. No real-image subpixel accuracy was achieved. Controlled same-image subpixel recovery was achieved.

## What changed

Created `src/lunarreg/` (15 modules), `src/run_lunarreg.py`, `tests/test_core.py`, `README.md`, `docs/validation.md`, `pyproject.toml`, `requirements-validated.txt`, and `.gitignore`. Modules cover IO, configuration, runtime/device handling, RoMa matching, cycle consistency, weighted geometry, multi-hypothesis/locked global refinement, local refinement, evaluation, benchmark/rejection checks, visualization, orchestration and CLI. Legacy scripts and V8/V9/V10 outputs were not edited or deleted. No dependencies were installed, no repository was published, and no new model weights were downloaded.

New safeguards include exclusive output-directory creation, deterministic seeds, image and transform validation, original-point confidence/cycle storage, input/transform hashes, explicit CPU retry for accelerator errors, stage logs, failure JSON, geometric gate reporting and JSON/CSV metrics. GeoTIFF evaluation is isolated after the image-only estimator. Spherical equirectangular evaluation preserves unwrapped lunar longitude and uses float64 coordinates. Existing outputs are never silently reused or overwritten.

## V11 outcome

The existing RoMa checkpoint exposes a trained fine-feature network. V11 loads only those weights; its stride-2 representation provides finer spatial support than DINO’s stride-16 tokens. It searches ±18 LRO pixels using normalized learned patch correlation, 13/21-pixel patch agreement, quadratic subpixel peaks, a peak-separation threshold, reverse consistency and compact support from neighbouring controls. The global affine is frozen. Thresholds were fixed before metadata evaluation.

On pair_002: **0/320 accepted**; 291 weak-similarity rejections, 23 search-boundary rejections, 6 border rejections. Accepted coverage and changed-pixel fraction are both 0%. V10 also accepted 0/320. V9, V10 and the V11 candidate therefore have the same final pixel registration. This is a negative refinement experiment, not an accuracy pass.

## Pair_002 results

Internal strict fit: **3.314117 px RMSE**, 26.5625% source-grid coverage. Relaxed control coverage: 29.6875%. The following use the same frozen transform and post-hoc metadata evaluation. They are metadata agreement, not surveyed landmark errors.

| Evaluation pool / convention | N | Median px | Mean px | P90 px | RMSE px | ≤0.5 | ≤1 | ≤2 | ≤5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| controls / legacy_corner | 320 | 6.2275 | 6.4696 | 8.3416 | 6.6301 | 0.00% | 0.00% | 0.00% | 10.62% |
| controls / pixel_center | 320 | 5.9164 | 6.1524 | 8.0220 | 6.3191 | 0.00% | 0.00% | 0.00% | 20.62% |
| cached_sample / legacy_corner | 4982 | 8.4147 | 8.8030 | 13.6311 | 9.3856 | 0.00% | 0.00% | 0.00% | 12.44% |
| cached_sample / pixel_center | 4982 | 8.0698 | 8.4676 | 13.3008 | 9.0677 | 0.00% | 0.00% | 0.00% | 16.82% |
| uniform_grid / legacy_corner | 1521 | 9.4141 | 9.4071 | 13.8110 | 9.9508 | 0.00% | 0.00% | 0.00% | 9.99% |
| uniform_grid / pixel_center | 1520 | 8.8912 | 8.9046 | 13.4017 | 9.4995 | 0.00% | 0.00% | 0.00% | 15.26% |

The historical V9 headline was 8.419 px median / 8.821 mean / 13.682 P90 on an original sample that was not fully saved. The cached sample above reproduces the paired V10 evaluation pool; it must not be called the identical historical sample. A uniform grid samples different regions and produces different metrics.

## Fresh inference limitation

Fresh seeded RoMa fast/bidirectional inference completed on MPS and passed the internal geometric gate (strict RMSE 3.2505 px, strict coverage 25.00%, relaxed coverage 31.25%). However, its same-grid legacy-corner metadata median was **18.3369 px**, versus **9.4141 px** for frozen V9. This run is retained as a **failed independent-validation result**, not promoted.

The refactor uses float32 descriptor inference on MPS/CPU for compatibility; old RoMa defaults used mixed precision. Sampling and backend numerics can change the chosen mode. The current geometric gate cannot reliably distinguish an accurate solution from a coherent but displaced match mode on this real pair. Do not use fresh inference as an automatically approved replacement for frozen V9. Calibrating that gate requires additional independent real pairs/landmarks. No GeoTIFF coordinates were fed back into fitting or thresholds to repair the failed result.

## Controlled benchmark

Exact seeded pixel-center transforms of the LRO image resized to 384×384. Each case evaluates 144 points over 56.25% of an 8×8 source grid (an interior grid with border margins). Coarse matching is actual RoMa fast/bidirectional plus V9-style global fitting, not ground-truth-correspondence fitting. Truth is used only for evaluation. All six global geometric gates passed: 0/6 case failures.

| Case | Coarse median | Coarse RMSE | Coarse P90 | V11 median | V11 RMSE | V11 P90 |
|---|---:|---:|---:|---:|---:|---:|
| translation | 0.1103 | 0.1103 | 0.1138 | 0.0151 | 0.0195 | 0.0317 |
| rotation | 0.0170 | 0.0171 | 0.0183 | 0.0147 | 0.0176 | 0.0257 |
| scale | 0.0195 | 0.0197 | 0.0232 | 0.0124 | 0.0154 | 0.0222 |
| affine | 0.0337 | 0.0356 | 0.0503 | 0.0156 | 0.0217 | 0.0338 |
| gamma | 0.0060 | 0.0061 | 0.0066 | 0.0162 | 0.0296 | 0.0467 |
| illumination | 0.0327 | 0.0344 | 0.0470 | 0.0361 | 0.0621 | 0.1044 |

**Both methods: 100% of 864 evaluated points ≤0.5, ≤1, ≤2 and ≤5 px.** V11 improves some cases but worsens rotation RMSE, gamma and illumination. It is not a universally beneficial default.

| Aggregate method | N | Median | RMSE | P90 | Maximum |
|---|---:|---:|---:|---:|---:|
| coarse_v9_style | 864 | 0.0206 | 0.0505 | 0.1092 | 0.1146 |
| v11_supported_field | 864 | 0.0156 | 0.0320 | 0.0435 | 0.2448 |

The isolated local-capture benchmark is separately reported in the CSV: rejected controls keep their identity prediction. In the illumination case it rejects 12/144 (8.33%); accepted-only RMSE is 0.1637 px, but all-point RMSE is 1.2657 px. This illustrates why accepted-only subpixel statistics must include rejection rates. Negative learned-descriptor checks rejected all 64 unrelated-noise and all 64 textureless controls (0 false accepts in these two tests).

## Tests and preservation

**18 tests passed** in the existing `.venv` using unittest. Tests cover weighted similarity/affine recovery, outlier-tolerant global recovery, affine bounds/reflections/degeneracy, transform application, cycle units, metric boundaries, local synthetic subpixel recovery, unrelated/ambiguous/border rejection, unsupported fields, 128-channel matching, lunar projection precision, invalid inputs and output preservation. Full MPS matching and controlled image runs passed execution; full CPU RoMa inference was stopped for runtime efficiency, so only CPU helper/local tests are claimed.

All 13 V9 files match SHA-256 values saved by V10. The production matrix and registered pixels exactly equal saved V9; V11 pixels also equal V9. V8/V10 files were only read. The early `production_v9_v11_20260907` evaluation used float32 metadata coordinates and is superseded by `production_final_20260907`; it is preserved, not overwritten. Development attempts remained in the task work directory.

## Outputs

- Recommended run: `outputs/pair_002/production_final_20260907/` (transform, registered/overlay/difference images, V11 field/control CSV, metrics JSON/CSV, config, logs, controls).
- Fresh inference, not promoted: `outputs/pair_002/production_fresh_20260907/` (full match confidence/cycle arrays, hypotheses/refinement logs and diagnostics).
- Benchmark: `outputs/benchmark_20260907/` (six image cases, exact transforms, evaluation arrays, per-control diagnostics, metrics JSON/CSV and negative checks).

## Recommended command

Run from any shell; change the final output suffix for subsequent runs:

```sh
cd /Users/safarhashim/Desktop/LunarReg && .venv/bin/python src/run_lunarreg.py register --source data/test_pairs/pair_002/chandrayaan.png --reference data/test_pairs/pair_002/lro.png --source-geotiff data/test_pairs/pair_002/chandrayaan.tif --reference-geotiff data/test_pairs/pair_002/lro.tif --frozen-baseline outputs/pair_002 --no-local-refine --output outputs/pair_002/production_replay_01
```

Benchmark reproduction:

```sh
cd /Users/safarhashim/Desktop/LunarReg && .venv/bin/python src/run_lunarreg.py benchmark --source data/test_pairs/pair_002/lro.png --seed 42 --output outputs/benchmark_replay_01
```

## Remaining limitations

No surveyed real-image subpixel evidence; one real pair and one controlled lunar source; no calibration across sensors, cast-shadow changes, terrain parallax or acquisition conditions. The quality gate is geometric support, not absolute correctness. Fresh inference is not yet reliable enough for unattended replacement of the verified baseline. V11’s compact support and interpolation are experimental. New machines require explicit dependency/model setup. This work does not claim general production certification beyond the preserved pair_002 baseline.

## Created source/document files

- `.gitignore`
- `README.md`
- `docs/validation.md`
- `pyproject.toml`
- `requirements-validated.txt`
- `src/lunarreg/__init__.py`
- `src/lunarreg/__main__.py`
- `src/lunarreg/benchmark.py`
- `src/lunarreg/config.py`
- `src/lunarreg/cycle.py`
- `src/lunarreg/evaluation.py`
- `src/lunarreg/geometry.py`
- `src/lunarreg/global_refinement.py`
- `src/lunarreg/io.py`
- `src/lunarreg/local_refinement.py`
- `src/lunarreg/matching.py`
- `src/lunarreg/pipeline.py`
- `src/lunarreg/runtime.py`
- `src/lunarreg/validation.py`
- `src/lunarreg/visualization.py`
- `src/run_lunarreg.py`
- `tests/test_core.py`
