"""Start both API processes with one command."""
from __future__ import annotations

import subprocess
import sys
import signal
import time

from server.config import API_HOST, FRONTEND_PORT, ROVER_PORT


def main() -> None:
    commands = [
        ("frontend", FRONTEND_PORT, "server.main:app"),
        ("rover", ROVER_PORT, "server.rover_api:app"),
    ]
    processes = [subprocess.Popen([
        sys.executable, '-m', 'uvicorn', module,
        '--host', API_HOST, '--port', str(port), '--log-level', 'info',
    ]) for _, port, module in commands]
    try:
        while any(process.poll() is None for process in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        for process in processes:
            process.send_signal(signal.SIGTERM)
        for process in processes:
            process.wait(timeout=5)


if __name__ == '__main__':
    main()
