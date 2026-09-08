"""Validated wire models for the rover API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExtensibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class IMU(ExtensibleModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class Odometry(ExtensibleModel):
    left_distance: float
    right_distance: float


class RangeReading(ExtensibleModel):
    altitude: float


class Telemetry(ExtensibleModel):
    rover_id: str = Field(min_length=1, max_length=128)
    frame_id: str = Field(min_length=1, max_length=128)
    timestamp: float
    imu: IMU
    odometry: Odometry
    range: RangeReading


class Heartbeat(ExtensibleModel):
    rover_id: str = Field(min_length=1, max_length=128)
    timestamp: float
    status: str = Field(min_length=1, max_length=32)


class Frame(ExtensibleModel):
    rover_id: str = Field(min_length=1, max_length=128)
    frame_id: str = Field(min_length=1, max_length=128)
    timestamp: float
    content_type: str = Field(default="image/jpeg", max_length=128)
    payload: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CommandResult(ExtensibleModel):
    rover_id: str = Field(min_length=1, max_length=128)
    command_id: str = Field(min_length=1, max_length=128)
    accepted: bool
    status: str = Field(min_length=1, max_length=64)
    message: str | None = None
