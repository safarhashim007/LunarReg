# OHRC held-out certification

Registration consumes the two PNG images only. Never pass GeoTIFF arguments,
scale/rotation priors, or frozen TMC transforms to a blind run. All twelve 30-degree
orientations are matched; candidates are mapped to original source coordinates
and pooled for image-only hypothesis selection and bounded affine fitting.
`transform_frozen.json` binds `transform.npy` by SHA-256 before evaluation.
Blind matching converts RoMa edge-origin coordinates to zero-based centers
before rotation mapping and fitting; legacy V9 coordinates are unchanged.
Historical outputs must never be reused as writable output directories.

The separate `lunarreg.postfreeze` command verifies the freeze before opening
controls. It evaluates every control, including failures and out-of-frame
predictions, without trimming or fitting a correction. CSV coordinates are
zero-based pixel centers. Errors measure ISRO/LRO mapping agreement, not surveyed
absolute ground truth. Reported percentages use strict inequalities.

Source-equivalent errors use a local 9-neighbor control-coordinate Jacobian,
computed only after freezing. This handles anisotropy without a misleading scalar
scale conversion. Conversion is invalid if any local linearization RMSE exceeds
0.1 reference pixels or its condition number reaches 100, or the displacement exceeds the local
reference support radius. Units refer to the
source PNG, not native OHRC detector pixels. Large displacements remain only a
first-order equivalent, not a physical relocation or recovered-detail claim.

The declared subpixel gate is P90 < 1 pixel in both reference and valid source
pixel-equivalent units. Median <1 alone does not pass. Production additionally
requires the image solver gate, at least 30 inliers and a 1% pooled-match ratio
within 5 reference pixels, at least 25% grid coverage in each image, the full
rotation sweep, input and frozen-output integrity, all 14,520 held-out controls,
full regression tests, and two exact same-seed fit replays. Neural inference
repeatability is explicitly not certified by fit replay. These gates must not be
changed after inspecting held-out errors to promote this pair.

Run from `/Users/safarhashim/Desktop/LunarReg`:

```sh
cd /Users/safarhashim/Desktop/LunarReg
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m lunarreg.postfreeze --run outputs/pair_ohrc_001_blind_006_cpu --controls data/test_pairs/pair_ohrc_001/geometry_controls_EVALUATION_ONLY.csv --output outputs/pair_ohrc_001_blind_006_cpu/heldout_evaluation.json
PYTHONPATH=src .venv/bin/python -m lunarreg.readiness --run outputs/pair_ohrc_001_blind_006_cpu --evaluation outputs/pair_ohrc_001_blind_006_cpu/heldout_evaluation.json --regression work/completion_audit/regression-final.log --output outputs/pair_ohrc_001_blind_006_cpu/production_readiness.json
```

The evaluator and readiness report refuse to overwrite existing reports.
Missing, invalid, or uncompleted evidence fails readiness; no numeric metrics
may be invented for an unfinished run.
