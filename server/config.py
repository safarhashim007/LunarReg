"""Environment-backed API configuration."""
from __future__ import annotations

import os


API_VERSION = "0.1.0"
API_HOST = os.getenv("LUNARREG_API_HOST", "0.0.0.0")
ROVER_PORT = int(os.getenv("LUNARREG_ROVER_PORT", "8001"))
FRONTEND_PORT = int(os.getenv("LUNARREG_FRONTEND_PORT", "8000"))


def frontend_origins() -> list[str]:
    value = os.getenv("LUNARREG_FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [origin.strip() for origin in value.split(",") if origin.strip()]
