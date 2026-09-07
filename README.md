# LunarReg

Registers overlapping lunar images using image correspondences. The production global method is RoMa v2 fast/bidirectional → reverse cycle checks → multi-hypothesis selection → locked weighted similarity refinement → constrained affine refinement (V9). GeoTIFF metadata is opened only after fitting and optional local candidates finish.

**Current deployment boundary:** use the hash-verified frozen replay for pair_002. Fresh real-pair inference passed the geometric gate but worsened independent metadata agreement; it remains a candidate requiring validation. The package is not certified for unattended new acquisitions. See `docs/validation.md` for the failed fresh run as well as successful controlled results.

V11 is experimental: pretrained RoMa fine-network descriptors at stride 2, bounded ±18 reference-pixel search, two patch sizes, subpixel quadratic peak interpolation, ambiguity rejection, mutual cycle checks, and compact spatial support. It never refits the frozen global transform. The default output remains V9; `v11_*` artifacts are separate experimental candidates.

## Setup and usage

Use the existing Python 3.12 `.venv`; no new dependencies were installed. `requirements-validated.txt` records the installed versions. On a new system, install the project dependencies and explicitly cache the pinned RoMa/DINO weights according to their upstream setup. The CLI does not initiate model downloads when caches are absent. Model/data redistribution and licensing must be checked separately before publishing assets.

From the repository root:

```sh
.venv/bin/python src/run_lunarreg.py register --source data/test_pairs/pair_002/chandrayaan.png --reference data/test_pairs/pair_002/lro.png --source-geotiff data/test_pairs/pair_002/chandrayaan.tif --reference-geotiff data/test_pairs/pair_002/lro.tif --frozen-baseline outputs/pair_002 --no-local-refine --output outputs/pair_002/production_replay_01
```

This recommended pair_002 command explicitly replays the preserved V9 transform and verifies source/reference hashes against the original pair. Use a **new output directory each time**; existing directories are refused. Omit `--frozen-baseline` for fresh RoMa registration of any supplied image pair. Add `--experimental-local` to evaluate V11 separately. `--no-local-refine` overrides that flag. GeoTIFF arguments are optional and must be supplied together.

Other options: `--seed 42`, `--mode fast|base|precise`, `--device auto|cpu|mps|cuda`, `--expected-scale`, `--expected-rotation`. Fast is the validated mode. Priors assume similar footprints and approximately north-up images; for different acquisitions, specify image-independent scale/orientation priors explicitly. Do not derive tuning values from evaluation landmarks.

```sh
.venv/bin/python src/run_lunarreg.py benchmark --source data/test_pairs/pair_002/lro.png --seed 42 --output outputs/benchmark_replay_01
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Architecture

- `io.py`, `config.py`, `runtime.py`: validation, immutable output namespaces, defaults, seeds, cached models and device handling.
- `matching.py`, `cycle.py`: forward sampling, reverse queries and confidence/cycle calculations.
- `geometry.py`, `global_refinement.py`: weighted geometry, image-only hypothesis scoring and mode-locked refinement.
- `local_refinement.py`: experimental learned-descriptor local search and supported displacement fields.
- `evaluation.py`, `benchmark.py`: post-hoc metadata agreement and exact controlled pixel transforms.
- `visualization.py`, `pipeline.py`, `__main__.py`: image diagnostics, orchestration and CLI.

Legacy scripts remain untouched and retain their historical behavior, including fixed pair paths and overwrite behavior. Use the new entry point for protected outputs. No historical experiment outputs were deleted.

## Accuracy and limits

These are different measurements:

1. **Internal fit RMSE:** residuals to selected image matches; a good fit does not establish correct landmarks.
2. **GeoTIFF agreement:** discrepancy against metadata-derived coordinates, not surveyed lunar landmark truth. Both legacy corner and pixel-center conventions are reported. This pair uses unwrapped lunar longitude near 340°; the evaluator preserves that branch for compatible spherical equirectangular metadata instead of allowing GDAL to wrap it to −20°.
3. **Controlled accuracy:** exact synthetic transformations of one lunar source image, reported in reference pixels. It measures recovery under the simulated warps and photometric changes, not the full cross-sensor lunar domain.

Historical V9: internal strict RMSE 3.314 px, strict coverage 26.56%; historical sampled metadata median 8.419 px, mean 8.821 px, P90 13.682 px. That original sample was not fully saved, so its exact sample statistics cannot be regenerated. V10 accepted 0/320 controls and retained identical V9 pixels. Current results are recorded in `docs/validation.md` and output JSON/CSV files.

The quality gate measures geometric support, coverage and prior agreement; it does not certify subpixel accuracy. Seeds are set for Python, NumPy, OpenCV and Torch. Backend/version changes can still alter matches; input and transform hashes distinguish frozen replay from fresh inference. Automatic accelerator failure retries global inference on CPU; local errors are reported explicitly. Out-of-support local regions remain zero. Experimental fields are not promoted automatically.

The benchmark has six seeded cases: translation, rotation, scale, mild affine, gamma and smooth illumination/noise. It reports coarse global and local refinement separately, plus isolated local capture, accepted-only and all-point metrics, coverage, rejection and quality-gate failure. It does not model cast-shadow reversal, terrain parallax, sensor PSFs or uncertain real georeferencing. Broader independent real pairs and surveyed controls remain necessary.

## Experiment history

SIFT → RoMa configuration ablations → V7 multi-hypothesis mode selection → V8 locked robust similarity → V9 constrained affine → V10 negative gradient/phase refinement → V11 learned fine-feature refinement. Historical scripts and artifacts remain the record of each experiment. The current refactor preserves V9 stage order, makes priors configurable, rejects degenerate affine fits and saves all new match quality arrays with their original points.

## Outputs and failure behavior

Runs save configuration/provenance, transform, diagnostics, stage logs, match/control arrays, JSON/CSV metrics and optional V11 fields/control rejection reasons. Exit 0 means the geometric gate passed; 2 means it failed; 1 indicates a runtime/input failure with `failure.json`. A failed gate still preserves diagnostics and must not be treated as registration approval. Source images, model caches, virtual environments and generated outputs are ignored by Git. The source repository is published at https://github.com/safarhashim007/LunarReg.

## Repository contents and change record

See [the change record](docs/changes.md) for the initial source snapshot and
[validation notes](docs/validation.md) for recorded experiments, measurements,
and limitations. This initial commit captures existing work; it does not imply
that historical experiments were rerun when the repository was committed.

| Path | Purpose |
|---|---|
| `.gitignore` | Excludes environments, caches, data, models and generated outputs. |
| `README.md` | Setup, commands, architecture and accuracy boundaries. |
| `default.profraw` | Empty profiling artifact retained explicitly in the initial snapshot. |
| `docs/` | Validation evidence and documented changes. |
| `pyproject.toml` | Package metadata, dependencies and CLI entry point. |
| `requirements-validated.txt` | Versions recorded from the validation environment. |
| `src/` | Packaged pipeline, CLI and historical experiment scripts. |
| `tests/` | Automated regression tests. |

Data, weights and generated validation artifacts are not bundled in Git. Commands
that reference them require the corresponding local files and cached models.
