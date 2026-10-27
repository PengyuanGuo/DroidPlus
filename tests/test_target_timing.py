"""Offline server diagnostics: valid targets queue without calling robot.move."""
import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def test_target_handler_queues_without_executing_motion():
    path = Path(__file__).resolve().parents[1] / 'droid_plus/services/franky_service.py'
    spec = importlib.util.spec_from_file_location('_test_franky_timing', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'franky': SimpleNamespace(), spec.name: module}):
        spec.loader.exec_module(module)
        robot = Mock()
        state = module.AppState(robot=robot, lock=threading.Lock(), latest_target=None,
                                stop_latched=True, shutdown=threading.Event(), control_thread=None,
                                last_control_error=None, command_timeout_s=10.)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(franky_state=state)))
        payload = module.JointTargetIn(positions=[0.] * 7, velocities=[0.] * 7, seq=42)
        result = module.post_target_joint_state(payload, request)
        assert result.seq == 42 and not state.stop_latched
        assert state.target_timing['seq'] == 42
        assert state.target_timing['lock_wait_ms'] >= 0
        robot.move.assert_not_called()
        health = module.health(request)
        assert health['target_timing'] == state.target_timing


def test_target_middleware_reports_request_duration():
    import asyncio
    path = Path(__file__).resolve().parents[1] / 'droid_plus/services/franky_service.py'
    spec = importlib.util.spec_from_file_location('_test_franky_middleware', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'franky': SimpleNamespace(), spec.name: module}):
        spec.loader.exec_module(module)
        state = SimpleNamespace(target_timing={})
        request = SimpleNamespace(method='POST', url=SimpleNamespace(path='/target_joint_state'),
                                  app=SimpleNamespace(state=SimpleNamespace(franky_state=state)))
        response = SimpleNamespace(headers={}, status_code=422)

        async def next_handler(req):
            await asyncio.sleep(.002)
            return response

        result = asyncio.run(module.target_request_timing(request, next_handler))
        assert 'target_request;dur=' in result.headers['Server-Timing']
        assert state.target_timing['request_ms'] >= 2
        assert state.target_timing['status'] == 422
