# Franka Hand Teleop — Development Notes (2026-09-19)

Summary of changes that made SO-101 → Franka teleop work with the **Franka Hand** (no Robotiq). Companion to [teleop.md](teleop.md).

## Goal

Run:

```bash
python scripts/run_teleop_cli.py --gripper franka --no-record
```

with arm + Hand following the SO-101 leader, using the existing `GripperClient` HTTP API (Robotiq-style 0–255 bits).

## What Was Wrong / What We Hit


| Issue                                                 | Cause                                                        | Fix                                                                          |
| ----------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Wrong NUC IP (`192.168.50.129`)                       | Hardcoded in `teleop/bin/activate`                           | Point `NUC_IP` / `FRANKY_SERVICE_URL` at the live NUC (e.g. `192.168.9.167`) |
| FK / Z-safety crash when URDF missing                 | `enforce_min_z` called with `model=None`                     | No-op when FK unavailable                                                    |
| CLI ignored `--franky-service-url` for control        | Only used for URDF fetch                                     | Pass URL into `DroidPlus(franky_service_url=...)`                            |
| No Hand support                                       | Only Robotiq `gripper_service`                               | Franka Hand endpoints on `franky_service`                                    |
| `/activate` timed out at 5 s                          | Homing often takes 10–30 s                                   | Separate `activate_timeout_s` (60 s)                                         |
| Desk red “EE not connected” as soon as service starts | `franky.Gripper(ip)` held for process lifetime               | **Lazy connect** on `/connect`, **release** on `/disconnect`                 |
| Homing / motion faults (red Hand, “initializing”)     | `grasp()`, mid-move `stop()` spam, overlapping Moves         | Home once; Move-only; single in-flight Move                                  |
| Gripper lag / mid-stroke stutter                      | libfranka `move()` is point-to-point (must stop to retarget) | **Intent / binary follow modes** (not GELLO-style waypoint chaining)         |




## Architecture (Current)

```
SO-101 leader  ──teleop_runner──►  FrankyClient  ──HTTP──►  franky_service (NUC)
                                      │                         ├── Robot (arm)
                 bits 0=open..255=closed                         └── Gripper (Hand, lazy)
 GripperClient ──HTTP (same NUC URL)─────────────────────────────► /connect /activate /go_to …
```

- **Arm**: unchanged joint remapping (`so101_to_franka`).
- **Gripper**: same bit convention as Robotiq (`so101_gripper_to_robotiq`); Hand maps bits → finger width (m).
- Desk shows Hand **red while teleop holds the libfranka Gripper connection**; green again after `/disconnect` (teleop exit). That is expected, not a fault.



## Files Touched


| File                                    | Role                                                    |
| --------------------------------------- | ------------------------------------------------------- |
| `droid_plus/services/franka_gripper.py` | Bits ↔ width / speed helpers                            |
| `droid_plus/services/franky_service.py` | Hand HTTP API + follow loop + lazy connect              |
| `droid_plus/services/gripper_client.py` | Longer activate timeout; `disconnect()`                 |
| `droid_plus/datagen/setup.py`           | `init_gripper(..., backend=)`; Franka timeouts          |
| `droid_plus/datagen/teleop_runner.py`   | Stream gripper cmds without busy-wait                   |
| `droid_plus/datagen/safety.py`          | Skip Z clamp when FK model is `None`                    |
| `scripts/run_teleop_cli.py`             | `--gripper {robotiq,franka,none}`; finally → disconnect |
| `tests/test_franka_gripper.py`          | Bits/width round-trip                                   |




## Hand Control Design Decisions



### Keep from GELLO-style stacks

- Home **once** per session (`homing()`), then Move-only.
- Never use `grasp()` for teleop streaming (trips Desk / Hand errors).
- Measure / use real `max_width` after home.



### Do **not** copy GELLO’s wait-for-completion chaining for continuous squeeze

`Gripper.move(width, speed)` decelerates to a stop at each target. Retargeting requires `stop()`, which also kills velocity. Chaining intermediate leader widths → **visible mid-stroke halt**. GELLO mostly works because its leader is effectively binary.

### Follow modes (`GRIPPER_FOLLOW_MODE`)


| Mode               | Behavior                                                                                                          | When to use                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `intent` (default) | While leader is moving, one Move toward the extreme in that direction; on settle, `stop()` near the settled width | Continuous width tracking, smoother than waypoints |
| `binary`           | Hysteresis open/close                                                                                             | Most robust; matches policy binarization           |
| `waypoint`         | Idle-only Move to latest width                                                                                    | A/B / debug only (stutters on continuous input)    |


Env knobs (NUC):

```bash
# on the NUC
GRIPPER_FOLLOW_MODE=intent GRIPPER_SETTLE_S=0.15 GRIPPER_STOP_LEAD_M=0.004 make franky_service
```



## How to Run

1. Redeploy updated `franky_service` on the NUC (`FRANKA_GRIPPER=1`).
2. On the workstation:

```bash
export NUC_IP=192.168.9.167   # or set in teleop/bin/activate
python scripts/run_teleop_cli.py --gripper franka --no-record
```

1. Expect: homing ~10–30 s on first activate; Desk Hand red during the session; connection released on quit.

Manual checks:

```bash
curl -s http://$NUC_IP:54321/health
curl -s -X POST http://$NUC_IP:54321/connect
curl -s -X POST "http://$NUC_IP:54321/activate?wait=true"
curl -s -X POST http://$NUC_IP:54321/disconnect
```



## Known Limits

- **Desk red while connected** — libfranka owns the Hand; Desk cannot show it as connected at the same time.
- **~0.1 m/s Hand speed** — full ~80 mm stroke ≈ 0.8 s; fast SO-101 squeezes will always look slightly behind.
- **URDF** `/urdf` **404** — Z-safety disabled until the franky build exposes URDF (or set `FRANKA_URDF_PATH`); arm teleop still works.
- **Arm tracking** — still joint-space remapping (5→7 DoF), not Cartesian EE matching; slow `RelativeDynamicsFactor` on the NUC also adds lag.



## Quick Tuning Cheat Sheet


| Symptom                      | Try                                                      |
| ---------------------------- | -------------------------------------------------------- |
| Mid-stroke pause then resume | Prefer `binary`, or raise `GRIPPER_SETTLE_S` in `intent` |
| Overshoot past settled width | Raise `GRIPPER_STOP_LEAD_M`                              |
| Twitchy when leader still    | Raise `GRIPPER_DEADBAND_M`                               |
| Desk stays red after quit    | `curl -X POST http://$NUC_IP:54321/disconnect`           |


