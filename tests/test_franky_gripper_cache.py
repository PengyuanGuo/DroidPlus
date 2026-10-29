"""franky_service must not poll the Franka Hand width: each read stalls the process."""
import sys
import threading
import time
import types

import pytest

pytest.importorskip("fastapi")


class FakeGripper:
    def __init__(self, blocked_at=None):
        self.reads = 0
        self.moves = []
        self.blocked_at = blocked_at  # width where a grasped object stops the fingers
        self._width = 0.08
        self.max_width = 0.08

    @property
    def width(self):
        self.reads += 1
        return self._width

    def move(self, width, speed):
        self.moves.append(width)
        self._width = max(width, self.blocked_at) if self.blocked_at is not None else width
        return self.blocked_at is None

    def stop(self):
        pass


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setitem(sys.modules, "franky", types.ModuleType("franky"))
    sys.modules.pop("droid_plus.services.franky_service", None)
    from droid_plus.services import franky_service
    yield franky_service
    sys.modules.pop("droid_plus.services.franky_service", None)


def make_state(service, gripper):
    st = service.AppState(robot=None, lock=threading.Lock(), latest_target=None, stop_latched=True,
                          shutdown=threading.Event(), control_thread=None, last_control_error=None,
                          command_timeout_s=1.0)
    st.gripper, st.gripper_homed, st.gripper_max_width_m = gripper, True, 0.08
    return st


def request_for(st):
    return types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(franky_state=st)))


def test_gripper_state_is_cached_unless_live(service):
    g = FakeGripper()
    st = make_state(service, g)
    for _ in range(5):
        assert service.gripper_state(request_for(st), closed_threshold=128, live=False)["width_m"] == 0.08
    assert g.reads == 1  # First call fills the cache.
    service.gripper_state(request_for(st), closed_threshold=128, live=True)
    assert g.reads == 2


def test_finished_move_refreshes_cache_once(service):
    g = FakeGripper()
    st = make_state(service, g)
    service._issue_move(st, g, 0.0, 0.1, preempt=False)
    st.gripper_motion_thread.join(1)
    assert st.gripper_width_m == 0.0 and g.reads == 1 and not st.gripper_busy


def test_idle_follow_loop_does_not_poll_or_repress_grasped_object(service):
    g = FakeGripper(blocked_at=0.03)
    st = make_state(service, g)
    st.gripper_width_m = 0.08
    service._set_gripper_target(st, width_m=0.08, speed_m_s=0.1)
    service._set_gripper_target(st, width_m=0.0, speed_m_s=0.1)  # Close command.
    st.gripper_settle_s = 0.05
    loop = threading.Thread(target=service._gripper_command_loop, kwargs={"st": st}, daemon=True)
    loop.start()
    time.sleep(0.6)
    st.shutdown.set()
    loop.join(1)
    assert g.moves == [0.0]  # One close; no repeated Moves against the object.
    assert g.reads <= 2  # Refresh after the Move, not every 10 ms.
    assert st.gripper_width_m == pytest.approx(0.03)


def test_control_loop_survives_failed_recovery_and_reports_heartbeat(service):
    class FailingRobot:
        def recover_from_errors(self):
            raise RuntimeError("FCI not active")

    st = make_state(service, None)
    st.robot = FailingRobot()
    st.control_thread = threading.Thread(target=service._control_loop,
                                         kwargs={"st": st, "control_hz": 100, "timeout_s": 1.0}, daemon=True)
    st.control_thread.start()
    time.sleep(0.1)
    health = service.health(request_for(st))
    assert health["control_thread_alive"] and health["control_heartbeat_age_s"] < 0.5
    assert "FCI not active" in health["last_control_error"]
    st.shutdown.set()
    st.control_thread.join(1)
