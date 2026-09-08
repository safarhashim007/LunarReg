"""Read-only, allowlisted views over LunarReg output artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import HTTPException


PROJECT_ROOT = Path(os.environ.get('LUNARREG_ROOT', Path(__file__).resolve().parents[2])).resolve()
OUTPUTS_ROOT = (PROJECT_ROOT / 'outputs').resolve()


def safe_run_dir(run_name: str) -> Path:
    candidate = (OUTPUTS_ROOT / run_name).resolve()
    if candidate.parent != OUTPUTS_ROOT or not candidate.is_dir():
        raise HTTPException(status_code=404, detail='Run not found')
    return candidate


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def run_status(run_name: str):
    run = safe_run_dir(run_name)
    readiness = read_json(run / 'production_readiness.json')
    evaluation = read_json(run / 'heldout_evaluation.json')
    progress = read_json(run / 'rotation_progress.json')
    metrics = read_json(run / 'metrics.json')
    frozen = read_json(run / 'transform_frozen.json')
    config = read_json(run / 'config.json') or {}
    if readiness:
        state = 'production_ready' if readiness.get('production_ready') else 'failed_checks'
    elif frozen and evaluation:
        state = 'evaluated_pending_readiness'
    elif frozen:
        state = 'frozen_pending_evaluation'
    elif progress and not progress.get('complete', False):
        state = 'processing'
    elif metrics:
        state = 'completed_pending_freeze'
    else:
        state = 'not_started'
    return {
        'run': run_name,
        'state': state,
        'production_ready': bool(readiness and readiness.get('production_ready')),
        'rotation_progress': progress,
        'quality_gate': None if metrics is None else metrics.get('quality_gate'),
        'frozen': frozen is not None,
        'evaluation': evaluation,
        'readiness': readiness,
        'config': {key: config.get(key) for key in ('registration_mode', 'device', 'seed')},
        'available_files': sorted(p.name for p in run.iterdir() if p.is_file()),
    }
