# Hackathon demo

LunarReg is an image-registration pipeline for overlapping lunar imagery. The
demo shows a frozen V9 transform being replayed, validated, rendered, and
measured in a few minutes. It is intentionally separate from the unfinished
fresh OHRC blind sweep, so a live presentation never turns a partial experiment
into a scientific claim.

From `/Users/safarhashim/Desktop/LunarReg`:

```sh
chmod +x scripts/hackathon_demo.sh
scripts/hackathon_demo.sh
```

The script refuses to overwrite an existing output directory. It runs all
regression tests, replays the hash-verified V9 baseline for `pair_002`, and
creates `outputs/hackathon_demo_replay/` with:

- `overlay.png`: registered image over the LRO reference;
- `registered.png`: warped source image;
- `difference.png`: absolute visual difference;
- `metrics.json` and `metrics.csv`: machine-readable measurements;
- `transform_frozen.json`: the transform hash and image-only fitting receipt;
- `run.log`: the complete execution log.

For the stage explanation, show this sequence:

```text
source PNG + reference PNG
          ↓
RoMa correspondences + reverse-cycle checks
          ↓
weighted similarity + constrained affine fit
          ↓
freeze transform and record SHA-256
          ↓
optional post-freeze GeoTIFF agreement
          ↓
overlay, difference image, and metrics
```

The demo's frozen replay is the presentation baseline. It does not claim that
fresh cross-sensor OHRC inference is production-certified. The fresh run is
kept in `outputs/pair_ohrc_001_blind_006_cpu/`; its twelve-angle image-only
sweep must finish, freeze a transform, evaluate all 14,520 held-out controls,
and pass `lunarreg.readiness` before anyone calls it production-ready.

The judge-facing message is: “LunarReg separates choosing a transform from
checking it. The metadata is sealed during fitting, the transform is frozen by
hash, and every visual and numeric artifact is reproducible.”
# Mission Console UI

The repository now includes a small read-only UI for showing registration evidence.
It reads only the existing files under `outputs/`; it never fits a transform or
opens held-out geometry controls. A run is shown as `processing`, `frozen_pending_evaluation`,
`evaluated_pending_readiness`, `failed_checks`, or `production_ready` based on
which machine-readable artifacts exist.

Start it from the project root:

```sh
cd /Users/safarhashim/Desktop/LunarReg
PYTHONPATH=src python3 -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. The console lists runs with a `config.json`,
shows blind rotation progress, and exposes the readiness reasons. It does not
replace the post-freeze evaluator or the production gate; it makes their
evidence visible to a presenter or reviewer.
