"""Frontend-facing API routes."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from server.services.runs import OUTPUTS_ROOT, run_status
from server.storage.state import state


router = APIRouter()


@router.get('/system')
async def system():
    return {
        'backend': 'online',
        'device': 'unknown',
        'model': 'v12',
        'model_loaded': False,
        'active_runs': 0,
    }


@router.get('/rover')
async def rover():
    return state.frontend_rover()


@router.get('/telemetry')
async def telemetry():
    return {'items': state.frontend_telemetry()}


@router.get('/sensors')
async def sensors():
    return state.frontend_sensors()


@router.get('/runs')
async def runs():
    if not OUTPUTS_ROOT.is_dir():
        return {'runs': []}
    names = sorted(p.name for p in OUTPUTS_ROOT.iterdir() if p.is_dir() and (p / 'config.json').exists())
    return {'runs': names}


@router.get('/runs/{run_id}')
async def run(run_id: str):
    return run_status(run_id)


@router.get('/runs/{run_id}/metrics')
async def metrics(run_id: str):
    data = run_status(run_id)
    return {'run': run_id, 'metrics': data.get('evaluation'), 'quality_gate': data.get('quality_gate')}


@router.get('/runs/{run_id}/transform')
async def transform(run_id: str):
    data = run_status(run_id)
    run_dir = (OUTPUTS_ROOT / run_id).resolve()
    receipt = None
    receipt_path = run_dir / 'transform_frozen.json'
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
    return {'run': run_id, 'frozen': data['frozen'], 'receipt': receipt}


@router.get('/runs/{run_id}/artifacts')
async def artifacts(run_id: str):
    data = run_status(run_id)
    allowed = {
        'config.json', 'metrics.json', 'rotation_progress.json', 'rotation_sweep.json',
        'transform_frozen.json', 'heldout_evaluation.json', 'production_readiness.json',
        'registered.png', 'overlay.png', 'difference.png',
    }
    return {'run': run_id, 'artifacts': [name for name in data['available_files'] if name in allowed]}
