"""Frontend/Mission Console API application served on port 8000."""
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.api.frontend import router as frontend_router
from server.config import API_VERSION, frontend_origins
from server.services.runs import OUTPUTS_ROOT, run_status


UI_ROOT = (Path(__file__).resolve().parent / 'ui').resolve()

app = FastAPI(title='LunarReg Frontend API', version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins(),
    allow_credentials=False,
    allow_methods=['GET'],
    allow_headers=['*'],
)
app.mount('/ui', StaticFiles(directory=UI_ROOT), name='ui')


@app.get('/health')
async def health():
    return {
        'service': 'lunarreg-frontend-api',
        'status': 'ok',
        'version': API_VERSION,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


@app.get('/api/runs')
async def runs_compat():
    """Compatibility alias for the original Mission Console UI."""
    if not OUTPUTS_ROOT.is_dir():
        return {'runs': []}
    return {'runs': sorted(p.name for p in OUTPUTS_ROOT.iterdir() if p.is_dir() and (p / 'config.json').exists())}


@app.get('/api/runs/{run_name}')
async def run_status_compat(run_name: str):
    return run_status(run_name)


@app.get('/', include_in_schema=False)
async def index():
    return FileResponse(UI_ROOT / 'index.html')


@app.get('/api/rover/config')
async def legacy_rover_config():
    """Deprecated compatibility response; rover writes remain on port 8001."""
    return {'telemetry_interval_ms': 100, 'image_upload_enabled': True, 'sensor_upload_enabled': True}


app.include_router(frontend_router, prefix='/api/frontend')
