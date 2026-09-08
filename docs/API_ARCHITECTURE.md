# LunarReg API architecture

The main processing PC is the communication server. Both clients connect to
the PC through its Tailscale address; the frontend never connects directly to
the Raspberry Pi.

```text
                 MAIN PROCESSING PC
                 Tailscale interface
                         |
             +-----------+-----------+
             |                       |
        :8001 Rover API        :8000 Frontend API
             |                       |
       Raspberry Pi             Mission Console
             |                       |
       telemetry, frames,     processed run state,
       heartbeats, commands   metrics, artifacts
```

## Start both APIs

From the repository root:

```sh
cd /Users/safarhashim/Desktop/LunarReg
PYTHONPATH=src python3 -m server
```

The launcher starts two explicit Uvicorn processes:

- Rover/Pi API: `0.0.0.0:8001`
- Frontend/Mission Console API: `0.0.0.0:8000`

Ports and bind address can be changed with `LUNARREG_API_HOST`,
`LUNARREG_ROVER_PORT`, `LUNARREG_FRONTEND_PORT`, and
`LUNARREG_FRONTEND_ORIGINS`. The default frontend CORS allowlist is limited to
localhost development origins. The Rover API does not enable browser CORS.

## Tailscale access

Find the main PC address with:

```sh
tailscale ip -4
```

Use the returned address without committing it to the repository:

```text
Main Processing PC: 100.x.x.x
Frontend:           http://100.x.x.x:8000
Raspberry Pi:       http://100.x.x.x:8001
```

No public port forwarding, UPnP, or cloud tunnel is required or configured.

## Example requests

```sh
MAIN_PC_TAILSCALE_IP=100.x.x.x
curl "http://${MAIN_PC_TAILSCALE_IP}:8000/health"
curl "http://${MAIN_PC_TAILSCALE_IP}:8001/health"
curl "http://${MAIN_PC_TAILSCALE_IP}:8001/api/rover/config?rover_id=ROVER-001"
curl "http://${MAIN_PC_TAILSCALE_IP}:8000/api/frontend/system"
curl "http://${MAIN_PC_TAILSCALE_IP}:8000/api/frontend/rover"
```

Example Pi telemetry request:

```sh
curl -X POST "http://${MAIN_PC_TAILSCALE_IP}:8001/api/rover/telemetry" \
  -H 'Content-Type: application/json' \
  -d '{
    "rover_id":"ROVER-001",
    "frame_id":"FRAME-000001",
    "timestamp":1725781234.123,
    "imu":{"ax":0.01,"ay":-0.02,"az":9.81,"gx":0.001,"gy":-0.003,"gz":0.002},
    "odometry":{"left_distance":0.42,"right_distance":0.43},
    "range":{"altitude":2.35}
  }'
```

The API currently keeps recent telemetry and heartbeat state in process memory.
This is a deliberate foundation for a future durable storage adapter and job
queue; it does not change LunarReg registration or V12 checkpoint handling.

The frontend API exposes structured run summaries and an allowlisted artifact
list. It does not expose arbitrary filesystem paths, model checkpoints, shell
commands, or Python execution endpoints.
