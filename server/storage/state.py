"""Shared lightweight rover state store.

The two API processes need to observe the same heartbeat and telemetry state.
SQLite keeps this adapter local to the processing PC while avoiding a second
service. It stores only recent structured state; it never stores image payloads
or exposes filesystem paths.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any
from contextlib import contextmanager


DB_PATH = Path(os.environ.get('LUNARREG_API_STATE_DB', Path(__file__).resolve().parents[2] / 'outputs' / 'api_state.sqlite3'))
_lock = Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS heartbeats (rover_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS telemetry (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS frames (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS command_results (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)')
    return conn


@contextmanager
def _db():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


class RoverState:
    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _append(conn: sqlite3.Connection, table: str, payload: dict[str, Any]) -> None:
        conn.execute(f'INSERT INTO {table} (payload) VALUES (?)', (json.dumps(payload),))
        conn.execute(f'DELETE FROM {table} WHERE id NOT IN (SELECT id FROM {table} ORDER BY id DESC LIMIT 1000)')

    def heartbeat(self, data: dict[str, Any]) -> dict[str, Any]:
        received = {**data, 'server_timestamp': self.now()}
        with _lock, _db() as conn:
            conn.execute('INSERT OR REPLACE INTO heartbeats(rover_id,payload) VALUES (?,?)', (data['rover_id'], json.dumps(received)))
        return received

    def telemetry(self, data: dict[str, Any]) -> dict[str, Any]:
        received = {**data, 'server_timestamp': self.now()}
        with _lock, _db() as conn:
            self._append(conn, 'telemetry', received)
        return received

    def frame(self, data: dict[str, Any]) -> dict[str, Any]:
        received = {**data, 'server_timestamp': self.now()}
        received.pop('payload', None)
        with _lock, _db() as conn:
            self._append(conn, 'frames', received)
        return received

    def command_result(self, data: dict[str, Any]) -> dict[str, Any]:
        received = {**data, 'server_timestamp': self.now()}
        with _lock, _db() as conn:
            self._append(conn, 'command_results', received)
        return received

    def config(self, rover_id: str | None = None) -> dict[str, Any]:
        return {'rover_id': rover_id, 'telemetry_interval_ms': 100, 'image_upload_enabled': True, 'sensor_upload_enabled': True, 'queued_commands': 0}

    def commands(self, rover_id: str) -> list[dict[str, Any]]:
        return []

    def frontend_rover(self, rover_id: str = 'ROVER-001') -> dict[str, Any]:
        with _lock, _db() as conn:
            row = conn.execute('SELECT payload FROM heartbeats WHERE rover_id=?', (rover_id,)).fetchone()
        beat = json.loads(row[0]) if row else None
        return {'rover_id': rover_id, 'connection': 'online' if beat else 'offline', 'last_seen': beat.get('server_timestamp') if beat else None, 'status': beat.get('status') if beat else None}

    def frontend_telemetry(self, limit: int = 50) -> list[dict[str, Any]]:
        with _lock, _db() as conn:
            rows = conn.execute('SELECT payload FROM telemetry ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def frontend_sensors(self) -> dict[str, Any]:
        latest = self.frontend_telemetry(1)
        return {'latest': latest[0] if latest else None, 'available': bool(latest)}


state = RoverState()
