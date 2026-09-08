import unittest

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError as exc:  # API extras are optional in the scientific venv.
    raise unittest.SkipTest(f'API test dependencies are not installed: {exc}')

from server.config import FRONTEND_PORT, ROVER_PORT
from server.main import app as frontend_app
from server.rover_api import app as rover_app


TELEMETRY = {
    'rover_id': 'ROVER-001',
    'frame_id': 'FRAME-000001',
    'timestamp': 1725781234.123,
    'imu': {'ax': 0.01, 'ay': -0.02, 'az': 9.81, 'gx': 0.001, 'gy': -0.003, 'gz': 0.002},
    'odometry': {'left_distance': 0.42, 'right_distance': 0.43},
    'range': {'altitude': 2.35},
}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.rover = TestClient(rover_app)
        self.frontend = TestClient(frontend_app)

    def test_health_isolated(self):
        rover = self.rover.get('/health')
        frontend = self.frontend.get('/health')
        self.assertEqual(rover.status_code, 200)
        self.assertEqual(rover.json()['service'], 'lunarreg-rover-api')
        self.assertEqual(frontend.status_code, 200)
        self.assertEqual(frontend.json()['service'], 'lunarreg-frontend-api')

    def test_heartbeat_and_telemetry(self):
        heartbeat = self.rover.post('/api/rover/heartbeat', json={'rover_id': 'ROVER-001', 'timestamp': 1.0, 'status': 'online'})
        telemetry = self.rover.post('/api/rover/telemetry', json=TELEMETRY)
        self.assertTrue(heartbeat.json()['accepted'])
        self.assertTrue(telemetry.json()['accepted'])
        self.assertEqual(self.frontend.get('/api/frontend/rover').json()['connection'], 'online')
        self.assertGreaterEqual(len(self.frontend.get('/api/frontend/telemetry').json()['items']), 1)

    def test_invalid_telemetry_is_rejected(self):
        invalid = dict(TELEMETRY)
        invalid['imu'] = {'ax': 0.01}
        self.assertEqual(self.rover.post('/api/rover/telemetry', json=invalid).status_code, 422)

    def test_frontend_system_is_truthful(self):
        system = self.frontend.get('/api/frontend/system')
        self.assertEqual(system.status_code, 200)
        self.assertFalse(system.json()['model_loaded'])

    def test_ports_are_separate(self):
        self.assertEqual(ROVER_PORT, 8001)
        self.assertEqual(FRONTEND_PORT, 8000)
        self.assertNotEqual(ROVER_PORT, FRONTEND_PORT)

    def test_checkpoint_is_not_exposed(self):
        runs = self.frontend.get('/api/frontend/runs').json()['runs']
        if runs:
            artifacts = self.frontend.get(f'/api/frontend/runs/{runs[0]}/artifacts').json()['artifacts']
            self.assertFalse(any('checkpoint' in name.lower() or name.endswith(('.pt', '.pth', '.ckpt')) for name in artifacts))


if __name__ == '__main__':
    unittest.main()
