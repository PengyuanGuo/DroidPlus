"""franky_service must keep franky calls on one thread and avoid redundant preemptions."""
import sys
import threading
import time
import types

import pytest

pytest.importorskip("fastapi")


class FakeRobot:
    def __init__(self):
        self.moves = []
        self.state_reads = 0
        self.callers = set()

    @property
    def current_joint_state(self):
        self.state_reads += 1
        self.callers.add(threading.current_thread().name)
        return types.SimpleNamespace(position=[0.1] * 7, velocity=[0.0] * 7)

    def move(self, motion, asynchronous=False):
        self.callers.add(threading.current_thread().name)
        self.moves.append(motion)

    def recover_from_errors(self):
        pass


@pytest.fixture
def service(monkeypatch):
    franky = types.ModuleType("franky")
    franky.JointState = lambda q, dq: ("state", tuple(q), tuple(dq))
    franky.JointMotion = lambda target: ("motion", target)
    franky.JointStopMotion = lambda: ("stop",)
    monkeypatch.setitem(sys.modules, "franky", franky)
    sys.modules.pop("droid_plus.services.franky_service", None)
    from droid_plus.services import franky_service
    yield franky_service
    sys.modules.pop("droid_plus.services.franky_service", None)


def make_state(service, robot):
    return service.AppState(robot=robot, lock=threading.Lock(), latest_target=None, stop_latched=True,
                            shutdown=threading.Event(), control_thread=None, last_control_error=None,
                            command_timeout_s=10.0)


def request_for(st):
    return types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(franky_state=st)))


def target(service, q, seq=0):
    return service.LatestTarget(positions=list(q), velocities=[0.0] * 7, seq=seq,
                                accepted_timestamp_s=time.time())


def run_loop(service, st, seconds, hz=100):
    st.control_thread = threading.Thread(target=service._control_loop, name="control",
                                         kwargs={"st": st, "control_hz": hz, "timeout_s": 10.0}, daemon=True)
    st.control_thread.start()
    time.sleep(seconds)
    st.shutdown.set()
    st.control_thread.join(1)


def test_unchanged_target_is_not_resent_every_tick(service):
    robot = FakeRobot()
    st = make_state(service, robot)
    st.latest_target, st.stop_latched = target(service, [0.2] * 7), False
    run_loop(service, st, 0.3)  # ~30 ticks
    motions = [m for m in robot.moves if m[0] == "motion"]
    assert 1 <= len(motions) <= 3  # first send + periodic resend only


def test_changed_target_is_sent(service):
    st = make_state(service, None)
    now = time.monotonic()
    st.latest_target = target(service, [0.2] * 7)
    assert service._should_send(st, st.latest_target, now)
    import numpy as np
    st.last_sent = (np.full(7, 0.2), np.zeros(7), now)
    assert not service._should_send(st, st.latest_target, now + 0.01)
    assert service._should_send(st, target(service, [0.21] * 7), now + 0.01)
    assert service._should_send(st, st.latest_target, now + service.TARGET_RESEND_S)


def test_joint_state_endpoint_uses_control_thread_cache(service):
    robot = FakeRobot()
    st = make_state(service, robot)
    st.control_thread = threading.Thread(target=service._control_loop, name="control",
                                         kwargs={"st": st, "control_hz": 100, "timeout_s": 10.0}, daemon=True)
    st.control_thread.start()
    time.sleep(0.05)
    for _ in range(20):
        js = service.get_joint_state(request_for(st))
        assert js["positions"] == [0.1] * 7 and "stamp_s" in js
    st.shutdown.set()
    st.control_thread.join(1)
    assert robot.callers == {"control"}  # HTTP thread never touched the robot


def test_joint_state_returns_503_when_robot_is_wedged(service):
    robot = FakeRobot()
    st = make_state(service, robot)
    holder = threading.Thread(target=lambda: (st.robot_lock.acquire(), time.sleep(0.5), st.robot_lock.release()))
    holder.start()
    time.sleep(0.02)
    started = time.monotonic()
    with pytest.raises(service.HTTPException) as exc:
        service.get_joint_state(request_for(st))
    assert exc.value.status_code == 503
    assert time.monotonic() - started < 0.4
    holder.join()


def test_stop_latches_and_sends_stop_motion(service):
    robot = FakeRobot()
    st = make_state(service, robot)
    st.latest_target, st.stop_latched = target(service, [0.2] * 7), False
    service.stop(request_for(st))
    assert st.stop_latched and st.latest_target is None
    assert robot.moves == [("stop",)]
