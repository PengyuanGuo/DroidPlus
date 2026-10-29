# SpaceMouse teleoperation

Drive the Franka Panda with a USB SpaceMouse (tested: SpaceMouse Compact) instead of
the SO-101 leader arm. The SpaceMouse commands end-effector **XYZ translation and yaw**
in the robot base frame; roll/pitch are held at the starting tilt. Differential IK on
the workstation turns this into joint targets streamed to the existing
`franky_service` on the NUC, so recordings use the same 8-D joint+gripper
state/action format, cameras, labels and phase annotations as SO-101 teleop
(`meta.json` policy `teleop_spacemouse`).

## Status (2026-10-29)

| Area | State |
| --- | --- |
| Arm teleop, live | Working: ~99 Hz control, 4–10 ms NUC round trips, with and without the Franka Hand connected |
| Recording | Working: 15 Hz, left camera (wrist ZED currently unplugged) |
| Axis mapping | Verified on the rig: default `--axis-map=1,2,3,4,5,6` is correct |
| Gripper toggle | Not yet exercised live |
| Known issue | One `franky_service` hang during Hand `/connect`; restart cleared it (see Troubleshooting) |

## Setup

**Workstation** (SpaceMouse plugged in here): use `.venv`, which has `pyspacemouse`,
`pinocchio` and `viser` (`pip install -e '.[spacemouse,analysis]'`). For HID permissions
see the [PySpaceMouse docs](https://spacemouse.kubaandrysek.cz/).

```bash
source env_variables.md                                     # NUC_IP, camera serials
export FRANKA_URDF_PATH="$PWD/models/franka_panda/panda.urdf"  # NUC franky build has no /urdf
```

The arm is a **Panda** (the service docstring says FR3; ignore that), so the local
Panda URDF is correct. It ends at the flange (`panda_link8`).

**NUC**: run `franky_service` from this repo in the `droidplus` conda env. Before
starting it, stop any old instance; otherwise it fails with `[Errno 98] address
already in use` and the old code keeps serving.

```bash
ss -ltnp 'sport = :54321'     # find the old PID, then kill it
make franky_service           # keep your usual env vars, e.g. FRANKY_ROBOT_IP
```

`FRANKA_GRIPPER` defaults to `1`. To check the new service is live from the workstation:
`curl -s http://$NUC_IP:54321/openapi.json | grep -c '"live"'` must print at least 1.

## Workflow

```bash
# 1. Check axes and buttons (no robot): prints the --axis-map for where you stand
.venv/bin/python scripts/record_spacemouse.py record --guided
# 2. Practise on a virtual arm in viser, starting from the robot's pose (read-only)
.venv/bin/python scripts/spacemouse_viser.py --start nuc                  # http://localhost:8080
# 3. Dry run: reads robot state, sends no commands
.venv/bin/python scripts/run_spacemouse_cli.py --dry-run --no-record --gripper none --wrist-camera-id '' --viser
# 4. Live, recording
.venv/bin/python scripts/run_spacemouse_cli.py --gripper franka --wrist-camera-id '' --viser --task <task>
```

`run_teleop_cli.py` / `run_teleop_ui.py --controller spacemouse` also work (the UI has no `--viser`).
`--wrist-camera-id ''` skips the unplugged wrist camera; only cameras in use are waited for.

## Controls (default `--button-mode clutch`)

1. SPACE starts an episode. Center the cap and release both buttons to arm.
2. **Hold button 0** and push/pull the cap to translate, twist it for yaw. Release to hold the pose.
3. **While holding button 0, tap button 1** to toggle the gripper (close, then open).
4. ESC ends the episode; then answer the valid/success/score/notes prompts.

Speed scales with cap deflection. Alternative `--button-mode deoxys --gripper-button 0
--enable-button 1`: free motion after arming, hold left button to close, right button stops.

## Limits and safety

| Option | Default | Meaning |
| --- | --- | --- |
| `--linear-speed` | 0.10 m/s | Max translation speed |
| `--angular-speed` | 0.3 rad/s | Max yaw rate |
| `--joint-speed` | 0.5 rad/s | Per-joint cap (10 cm/s needs up to ~0.34 near home) |
| `--min-z` | 0.12 m | Flange height floor; Franka Hand fingertips are ~0.103 m lower (shared default with SO-101) |
| `--deadzone` | 0.1 | Cap dead zone |
| `--input-timeout` | 0.25 s | Max age of active (non-zero) input |
| `--axis-map` | `1,2,3,4,5,6` | Signed permutation: device x,y,z,roll,pitch,yaw to base x,y,z,rx,ry,rz |

The episode stops (robot stop is sent) on any of these:

- Stale non-zero input or a SpaceMouse read error. A held button on a centered cap
  sends no reports and is fine.
- A control interval over 250 ms.
- A target post in flight for over 0.5 s (2 s for the first post).
- A joint tracking error over 0.15 rad.

IK rejects targets beyond URDF joint limits (0.01 rad margin) or below `--min-z`, and
the status shows `target rejected`. The height floor is not collision checking.

## How it works

| File | Role |
| --- | --- |
| `droid_plus/datagen/spacemouse.py` | HID reader thread, axis mapping, damped-least-squares IK (`CartesianController`) |
| `droid_plus/datagen/spacemouse_runner.py` | Episode loop: 100 Hz control, recording, viser, diagnostics |
| `droid_plus/datagen/async_target.py` | Posts the latest joint target from a worker; control never waits on HTTP |
| `droid_plus/datagen/async_recording.py` | Camera and state recording worker (latest snapshot, no queue) |
| `droid_plus/datagen/spacemouse_viz.py` | `--viser` view: measured arm (grey), target (orange), Hand, input arrow, floor, status |
| `scripts/record_spacemouse.py` | Record raw input / guided axis check / offline `replay` through the IK |
| `scripts/spacemouse_viser.py` | Virtual-arm simulator; never commands the robot |

The status line prints once per second: controller state, `control` Hz, `record` Hz,
`posts/s`, `slowest_post_ms`, `state_get_ms`. Healthy values are about 99 Hz,
15 Hz, ~100/s, under 10 ms and about 5 ms.

**NUC gripper note:** reading the Franka Hand width blocks the whole `franky_service`
process for 30–100 ms. The service therefore caches the width after each Move or
homing, and never polls it. `/gripper_state` returns the cached value; `?live=true`
forces a read, so don't use that during teleop. Recorded gripper observations update
after each Move finishes.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Connection refused` to `localhost:54321` | `NUC_IP` unset: `source env_variables.md` |
| `SpaceMouse requires a valid URDF` | `export FRANKA_URDF_PATH=...panda.urdf` |
| 15 s camera wait | Pass `--wrist-camera-id ''` for unplugged cameras |
| Control ~9 Hz, `state_get_ms` ~200 with `--gripper franka` | NUC runs the old service (no width cache): restart it from this repo |
| `Read timed out` on `/connect`, then every endpoint times out | `franky_service` hung (seen once during Hand connect). On the NUC run `py-spy dump --pid <PID>` to capture where it is stuck, then `kill -9` and restart |
| Arm doesn't follow the cap | Button 0 must be held (clutch mode) |
