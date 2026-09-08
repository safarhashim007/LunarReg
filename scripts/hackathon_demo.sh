#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

demo_output="${1:-outputs/hackathon_demo_replay}"
if [[ -e "$demo_output" ]]; then
  echo "Refusing to overwrite existing demo output: $demo_output" >&2
  exit 2
fi

echo "[1/3] Running regression suite"
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -q

echo "[2/3] Replaying frozen V9 baseline"
PYTHONPATH=src .venv/bin/python src/run_lunarreg.py register \
  --source data/test_pairs/pair_002/chandrayaan.png \
  --reference data/test_pairs/pair_002/lro.png \
  --source-geotiff data/test_pairs/pair_002/chandrayaan.tif \
  --reference-geotiff data/test_pairs/pair_002/lro.tif \
  --frozen-baseline outputs/pair_002 \
  --no-local-refine \
  --output "$demo_output"

echo "[3/3] Demo artifacts"
printf '%s\n' \
  "$repo_root/$demo_output/overlay.png" \
  "$repo_root/$demo_output/registered.png" \
  "$repo_root/$demo_output/metrics.json"
