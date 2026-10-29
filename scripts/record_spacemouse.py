#!/usr/bin/env python3
"""Record raw SpaceMouse input and replay it offline through the teleop IK.

No robot or services are contacted. Use this to check axis directions, buttons
and dead zone before live SpaceMouse teleop (``run_spacemouse_cli.py``).

    Usage:
    python scripts/record_spacemouse.py record --guided
    python scripts/record_spacemouse.py record --duration 20 --axis-map=-2,1,3,4,5,6
    python scripts/record_spacemouse.py replay output/spacemouse_recordings/<file>.csv

``record --guided`` asks for one motion per robot base-frame direction and
derives the ``--axis-map`` for where you stand and how the device is placed.
``record`` writes a CSV of raw axes, buttons and the mapped base-frame twist.
``replay`` feeds that CSV through :class:`CartesianController` from the Franka
home pose using the local Panda URDF, and reports the resulting end-effector
motion and gripper commands.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time

import numpy as np

from droid_plus.datagen.safety import DEFAULT_MIN_EE_Z
from droid_plus.datagen.spacemouse import CartesianController, SpaceMouseConfig, axis_twist
from droid_plus.services.franky_client import HOME_POSITION

AXES = ("x", "y", "z", "roll", "pitch", "yaw")
TWIST = ("vx", "vy", "vz", "wx", "wy", "wz")
DEFAULT_URDF = os.path.join(os.path.dirname(__file__), "..", "models", "franka_panda", "panda.urdf")

# Guided prompts: (instruction, base-frame twist index, candidate raw axes).
# Translations compete only among raw x/y/z, since pushing the cap also tilts it.
GUIDED_STEPS = [
    ("push the cap toward robot +X: the way the gripper moves AWAY from the robot base", 0, (0, 1, 2)),
    ("push the cap toward robot +Y: the robot's LEFT (looking from the base toward the gripper)", 1, (0, 1, 2)),
    ("pull the cap UP (robot +Z)", 2, (0, 1, 2)),
    ("twist the cap COUNTER-CLOCKWISE seen from above (robot +yaw)", 5, (3, 4, 5)),
]


def config_from_args(args) -> SpaceMouseConfig:
    return SpaceMouseConfig(
        button_mode=args.button_mode, linear_speed=args.linear_speed, angular_speed=args.angular_speed,
        joint_speed=args.joint_speed, deadzone=args.deadzone, enable_button=args.enable_button,
        gripper_button=args.gripper_button, axis_map=tuple(int(x) for x in args.axis_map.split(",")),
    )


# ── Record ───────────────────────────────────────────────────────────────────

class RawReader:
    """Drain every HID report in a thread; keep only the latest raw state.

    ``pyspacemouse`` processes one report per ``read()``. Reading slower than
    the device sends (or not at all while prompting) fills the kernel buffer,
    which then drops new reports, including the release back to zero.
    """
    def __init__(self, device_name=None):
        import pyspacemouse

        self.device = pyspacemouse.open(
            device=device_name, axis_convention=pyspacemouse.AxisConvention.HID_Z_UP,
        )
        if not self.device:
            raise SystemExit("No SpaceMouse found; check USB connection and HID permissions")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = (np.zeros(6), (0, 0))
        self.reports = 0
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        previous_stamp = -1.0  # pyspacemouse's stamp before any report
        try:
            while not self._stop.is_set():
                state = self.device.read()
                if state is None or not self.device.connected:
                    raise RuntimeError("SpaceMouse disconnected")
                if state.t == previous_stamp:
                    self._stop.wait(0.001)  # Nothing pending.
                    continue
                previous_stamp = state.t
                axes = np.array([state.x, state.y, state.z, state.roll, state.pitch, state.yaw], dtype=float)
                with self._lock:
                    self._latest = (axes, tuple(int(b) for b in state.buttons))
                    self.reports += 1
        except Exception as exc:
            self._error = exc

    def read(self):
        if self._error is not None:
            raise RuntimeError("SpaceMouse read failed") from self._error
        with self._lock:
            return self._latest[0].copy(), self._latest[1], self.reports

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.device.close()


def _capture(reader, config, duration, rate_hz, writer, t_start, label=""):
    """Sample the latest state for ``duration`` seconds; return the rows written."""
    period = 1.0 / rate_hz
    rows = []
    end = time.monotonic() + duration
    last_print = 0.0
    while time.monotonic() < end:
        tick = time.monotonic()
        axes, buttons, reports = reader.read()
        twist = axis_twist(axes, config)
        row = [round(tick - t_start, 4), reports, *axes, *buttons[:2], *twist, label]
        writer.writerow(row)
        rows.append(row)
        if tick - last_print >= 0.2:
            last_print = tick
            sys.stdout.write(
                f"\r t={tick - t_start:6.2f}s raw=[{' '.join(f'{a:+.2f}' for a in axes)}] "
                f"btn={list(buttons)} xyz=[{' '.join(f'{v:+.3f}' for v in twist[:3])}] yaw={twist[5]:+.3f}   "
            )
            sys.stdout.flush()
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    sys.stdout.write("\n")
    return rows


def _wait_centered(reader, deadzone, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        axes, _, _ = reader.read()
        if np.max(np.abs(axes)) <= deadzone:
            return True
        time.sleep(0.01)
    print(f"  cap not centered after {timeout:.0f}s (raw={np.round(axes, 2)}); continuing")
    return False


def _dominant(rows, candidates):
    """Return ``(raw_index, sign, mean, consistency)`` for one guided step, or None."""
    axes = np.array([r[2:8] for r in rows], dtype=float)[:, candidates]
    active = axes[np.max(np.abs(axes), axis=1) > 0.3]
    if len(active) < 10:
        return None
    mean = active.mean(axis=0)
    k = int(np.argmax(np.abs(mean)))
    sign = 1 if mean[k] > 0 else -1
    # Fraction of active samples where this axis, with this sign, dominates.
    consistency = float(np.mean((np.argmax(np.abs(active), axis=1) == k) & (np.sign(active[:, k]) == sign)))
    return candidates[k], sign, mean, consistency


def _guided_report(rows_by_step, current_map):
    print("\nGuided axis check (robot base frame: +X away from base, +Y robot left, +Z up):")
    axis_map = [None] * 6
    ok = True
    for (instruction, base, candidates), rows in zip(GUIDED_STEPS, rows_by_step):
        found = _dominant(rows, candidates)
        if found is None:
            print(f"  [??] {TWIST[base]}: no sustained motion detected")
            ok = False
            continue
        raw, sign, mean, consistency = found
        means = " ".join(f"{AXES[c]}={m:+.2f}" for c, m in zip(candidates, mean))
        clear = consistency >= 0.8
        ok &= clear
        print(f"  [{'OK' if clear else '??'}] {TWIST[base]} <- {'+' if sign > 0 else '-'}raw {AXES[raw]}"
              f"  (mean {means}; {consistency:.0%} consistent)")
        axis_map[base] = sign * (raw + 1)
    used = [abs(a) for a in axis_map if a is not None]
    if not ok or len(set(used)) != 4:
        print("Mapping is ambiguous (mixed directions or repeated raw axis). Re-run, holding one clean motion per step.")
        return None
    # Roll/pitch are discarded by the controller; any remaining raw axes keep the map a permutation.
    axis_map[3], axis_map[4] = sorted(set(range(1, 7)) - set(used))
    result = ",".join(str(a) for a in axis_map)
    if tuple(axis_map) == tuple(current_map):
        print(f"Current --axis-map={result} already matches.")
    else:
        print(f"Suggested --axis-map={result}  (current: {','.join(str(a) for a in current_map)})")
    return result


def record(args):
    config = config_from_args(args)
    path = args.output or os.path.join(
        args.output_dir, time.strftime("spacemouse_%Y%m%d_%H%M%S.csv", time.localtime()))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    reader = RawReader(args.spacemouse_device)
    try:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "reports", *AXES, "button0", "button1", *TWIST, "label"])
            f.write(f"# axis_map={args.axis_map} deadzone={args.deadzone}\n")
            t_start = time.monotonic()
            if args.guided:
                print("Do each motion in the ROBOT's frame, from where you will stand while teleoperating.")
                rows_by_step = []
                for i, (instruction, _, _) in enumerate(GUIDED_STEPS, 1):
                    input(f"\nStep {i}/{len(GUIDED_STEPS) + 1}: {instruction}.\n"
                          f"  Press Enter, then do it and hold until the capture ends ({args.step_s:.0f} s)...")
                    _wait_centered(reader, config.deadzone)
                    rows_by_step.append(
                        _capture(reader, config, args.step_s, args.rate_hz, writer, t_start, instruction))
                input(f"\nStep {len(GUIDED_STEPS) + 1}/{len(GUIDED_STEPS) + 1}: buttons. "
                      "Press Enter, release the cap, then press and release button 0, then button 1...")
                buttons = _capture(reader, config, args.step_s + 1, args.rate_hz, writer, t_start, "buttons")
                _guided_report(rows_by_step, config.axis_map)
                for b in (0, 1):
                    states = [r[8 + b] for r in buttons]
                    presses = sum(1 for prev, cur in zip([0] + states, states) if cur and not prev)
                    print(f"  [{'OK' if presses else '??'}] button {b}: {presses} press(es)")
            else:
                print(f"Recording {args.duration:.0f}s at {args.rate_hz:.0f} Hz (Ctrl-C to stop early)...")
                try:
                    _capture(reader, config, args.duration, args.rate_hz, writer, t_start)
                except KeyboardInterrupt:
                    print("\nStopped.")
            elapsed = time.monotonic() - t_start
    finally:
        reader.close()
    print(f"Device reports: {reader.reports} over {elapsed:.1f}s")
    print(f"Saved {path}")
    print(f"Replay: python scripts/record_spacemouse.py replay {path} --axis-map=<map>")


# ── Replay ───────────────────────────────────────────────────────────────────

def _load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    if not rows:
        raise SystemExit(f"No samples in {path}")
    t = np.array([float(r["t"]) for r in rows])
    axes = np.array([[float(r[a]) for a in AXES] for r in rows])
    buttons = np.array([[int(r["button0"]), int(r["button1"])] for r in rows])
    return t, axes, buttons


def replay(args):
    from droid_plus.analysis.end_effector_pose import build_model_from_urdf

    config = config_from_args(args)
    with open(args.urdf) as f:
        model, data, ee_frame = build_model_from_urdf(f.read())
    t, axes, buttons = _load(args.path)
    if max(config.enable_button, config.gripper_button) >= buttons.shape[1]:
        raise SystemExit("Configured button index is not in the recording")
    q0 = np.array(HOME_POSITION, dtype=float)
    ctrl = CartesianController(model, data, ee_frame, q0, config, args.min_z)
    import pinocchio as pin

    def ee(q):
        pin.framesForwardKinematics(model, data, ctrl.full_q(q))
        pose = data.oMf[ctrl.frame]
        return pose.translation.copy(), pose.rotation.copy()

    p0, r0 = ee(q0)
    traj, events, statuses = [], [], {}
    previous = t[0] - 1.0 / args.rate_hz
    for i in range(len(t)):
        twist = axis_twist(axes[i], config)
        enabled = bool(buttons[i, config.enable_button])
        grip = bool(buttons[i, config.gripper_button])
        if config.button_mode == "deoxys":
            if enabled:
                events.append((t[i], "stop button: episode would end"))
                break
            enabled = True
        dt = max(t[i] - previous, 1e-3)
        if dt > 0.25:  # Guided recordings pause at prompts; that is not a control stall.
            events.append((t[i], f"recording gap {dt:.1f}s skipped"))
            dt = 1.0 / args.rate_hz
        previous = t[i]
        q, gripper_percent = ctrl.step(twist, enabled, grip, ctrl.q, dt)
        statuses[ctrl.status] = statuses.get(ctrl.status, 0) + 1
        if gripper_percent is not None:
            events.append((t[i], "gripper CLOSE" if gripper_percent == 0.0 else "gripper OPEN"))
        p, _ = ee(q)
        traj.append([t[i], *q, *p])

    traj = np.array(traj)
    p1, r1 = ee(ctrl.q)
    yaw = np.degrees(pin.log3(r1 @ r0.T)[2])
    tilt = np.degrees(np.linalg.norm(pin.log3(r1 @ r0.T)[:2]))
    path_len = np.sum(np.linalg.norm(np.diff(traj[:, 8:11], axis=0), axis=1)) if len(traj) > 1 else 0.0
    print(f"Replayed {len(traj)} samples over {t[-1] - t[0]:.1f}s; EE frame {ee_frame!r}, min_z={args.min_z}")
    print(f"  start EE xyz = {np.round(p0, 3)}")
    print(f"  end   EE xyz = {np.round(p1, 3)}  (delta {np.round(p1 - p0, 3)} m, path {path_len:.3f} m)")
    print(f"  yaw change {yaw:+.1f} deg, tilt drift {tilt:.2f} deg")
    print(f"  EE range x[{traj[:, 8].min():.3f},{traj[:, 8].max():.3f}] "
          f"y[{traj[:, 9].min():.3f},{traj[:, 9].max():.3f}] z[{traj[:, 10].min():.3f},{traj[:, 10].max():.3f}]")
    print("  controller status counts: " + ", ".join(f"{k}={v}" for k, v in statuses.items()))
    for stamp, event in events:
        print(f"  t={stamp:6.2f}s {event}")
    if not ctrl.armed:
        print("  WARNING: controller never armed (cap was never centered with buttons released)")
    out = os.path.splitext(args.path)[0] + "_replay.csv"
    np.savetxt(out, traj, delimiter=",", comments="",
               header="t," + ",".join(f"q{j}" for j in range(7)) + ",ee_x,ee_y,ee_z")
    print(f"Saved joint/EE trajectory to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Record raw SpaceMouse input to CSV")
    rec.add_argument("--duration", type=float, default=20.0, help="Seconds to record (free mode)")
    rec.add_argument("--guided", action="store_true", help="Prompt for each axis and button, then derive --axis-map")
    rec.add_argument("--step-s", type=float, default=3.0, help="Seconds per guided step")
    rec.add_argument("--output-dir", default="output/spacemouse_recordings")
    rec.add_argument("--output", help="CSV path (default: timestamped in --output-dir)")
    rec.add_argument("--spacemouse-device", help="PySpaceMouse device name (default: auto-detect)")
    rep = sub.add_parser("replay", help="Replay a recording through the IK controller (no robot)")
    rep.add_argument("path")
    rep.add_argument("--urdf", default=os.path.normpath(DEFAULT_URDF))
    rep.add_argument("--min-z", type=float, default=DEFAULT_MIN_EE_Z)
    for p in (rec, rep):
        p.add_argument("--rate-hz", type=float, default=100.0)
        p.add_argument("--button-mode", choices=["clutch", "deoxys"], default="clutch")
        p.add_argument("--linear-speed", type=float, default=0.10)
        p.add_argument("--angular-speed", type=float, default=0.3)
        p.add_argument("--joint-speed", type=float, default=0.5)
        p.add_argument("--deadzone", type=float, default=0.1)
        p.add_argument("--enable-button", type=int, default=0)
        p.add_argument("--gripper-button", type=int, default=1)
        p.add_argument("--axis-map", default="1,2,3,4,5,6")
    args = parser.parse_args()
    record(args) if args.command == "record" else replay(args)


if __name__ == "__main__":
    main()
