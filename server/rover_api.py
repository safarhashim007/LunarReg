"""Rover API application served on port 8001."""
from datetime import datetime, timezone

from fastapi import FastAPI

from server.api.rover import router as rover_router
from server.config import API_VERSION


app = FastAPI(title="LunarReg Rover API", version=API_VERSION)


@app.get('/health')
async def health():
    return {
        'service': 'lunarreg-rover-api',
        'status': 'ok',
        'version': API_VERSION,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


app.include_router(rover_router, prefix='/api/rover')
