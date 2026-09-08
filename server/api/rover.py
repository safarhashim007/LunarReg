"""Rover-facing API routes."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from server.models.telemetry import CommandResult, Frame, Heartbeat, Telemetry
from server.storage.state import state


router = APIRouter()


def acknowledgement(rover_id: str, **extra):
    return {"accepted": True, "server_timestamp": datetime.now(timezone.utc).isoformat(), "rover_id": rover_id, **extra}


@router.post('/heartbeat')
async def heartbeat(payload: Heartbeat):
    state.heartbeat(payload.model_dump())
    return acknowledgement(payload.rover_id)


@router.get('/config')
async def config(rover_id: str | None = Query(default=None, max_length=128)):
    return state.config(rover_id)


@router.post('/telemetry')
async def telemetry(payload: Telemetry):
    state.telemetry(payload.model_dump())
    return acknowledgement(payload.rover_id, frame_id=payload.frame_id)


@router.post('/frame')
async def frame(payload: Frame):
    state.frame(payload.model_dump())
    return acknowledgement(payload.rover_id, frame_id=payload.frame_id, queued=False)


@router.get('/commands')
async def commands(rover_id: str = Query(..., min_length=1, max_length=128)):
    return {"rover_id": rover_id, "commands": state.commands(rover_id)}


@router.post('/command-result')
async def command_result(payload: CommandResult):
    state.command_result(payload.model_dump())
    return acknowledgement(payload.rover_id, command_id=payload.command_id)
