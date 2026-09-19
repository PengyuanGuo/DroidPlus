# Teleoperation

Teleoperate the Franka using an SO-101 leader arm. Records data in the same format as the experiment runners.

![Teleop walkthrough](images/teleop_video.mp4){width=100%}

## Prerequisites

### Hardware

- Franka robot arm, brought up (see [Hardware Setup](hardware-setup.md))
- [SO-101 leader arm](https://github.com/jess-moss/so-arm100) connected via USB
- Robotiq 2F-85 gripper (optional — use `--no-gripper` to skip)

### Software

```bash
# LeRobot 0.4.x with Feetech support and FK safety dependencies
cd /path/to/droid-plus
pip install -e ".[teleop,analysis]"
```

### Finding the SO-101 Serial Port

```bash
ls /dev/ttyACM*

# If multiple devices:
dmesg | grep ttyACM | tail -5
udevadm info --name=/dev/ttyACM0 --attribute-walk | grep -i 'manufacturer\|product'
```

The SO-101 typically shows up as `/dev/ttyACM0` (the default).

## Run

Bring up the Franka first (see [Bring-up procedure](../README.md#bring-up-procedure)), then use one of:

### CLI

```bash
python scripts/run_teleop_cli.py
python scripts/run_teleop_cli.py --port /dev/ttyACM1
python scripts/run_teleop_cli.py --no-gripper
python scripts/run_teleop_cli.py --no-record
python scripts/run_teleop_cli.py --task "pick_banana" --notes "first attempt"
```

#### Episode Flow (CLI)

1. **SPACE** — start an episode (begins recording + streaming)
2. **ESC** — stop the current episode
3. Post-episode prompts: `valid?`, `success?`, `score?`, `episode_notes?`
4. Loop back to step 1 for the next episode
5. **Ctrl+C** or ESC at the SPACE prompt — quit

Episodes are saved as `output/{task_}teleop_{timestamp}/episode_000/`, `episode_001/`, etc.

#### CLI Reference (`run_teleop_cli.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `/dev/ttyACM0` | SO-101 serial port |
| `--franky-service-url` | from `constants.py` | Franky service URL (URDF fetch) |
| `--no-gripper` | off | Skip gripper initialization |
| `--rate-hz` | `100.0` | Control loop rate (Hz) |
| `--min-z` | `0.23` | Min EE Z height in meters (table safety) |
| `--no-record` | off | Disable recording (records by default) |
| `--record-rate-hz` | `15.0` | Recording rate (Hz) — images captured at this rate |
| `--record-jpeg-quality` | `RECORD_JPEG_QUALITY` (constants.py) | JPEG quality for recorded images |
| `--output-dir` | `output` | Base output directory |
| `--task` | "" | Task name for metadata and output directory |
| `--notes` | "" | Free-form notes saved to metadata |
| `--dry-run` | off | Read SO-101 but don't command the robot or gripper |

### Web UI

```bash
python scripts/run_teleop_ui.py
python scripts/run_teleop_ui.py --port /dev/ttyACM1 --web-port 54325
python scripts/run_teleop_ui.py --output-dir runs --task pick_banana
```

The server binds to `0.0.0.0`; open `http://localhost:54325` (or `http://<host>:54325` from another machine on the same network) in a browser. The SO-101 leader is connected at launch, so the first episode is ready to start as soon as the page loads.

![Teleop Web UI](images/teleop_ui.png)

#### Page layout

1. **Header** — service name, status badge (idle / running), episode counter.
2. **Teleop card** — session settings resolved at launch: SO-101 port (read-only), control rate Hz, record rate Hz, recording output dir. Only `Min EE z` is editable between episodes.
3. **Episode card** — optional instruction, experiment / task names, max walltime (blank = unlimited).
4. **Status area** — live details while an episode is running (elapsed time, current instruction).
5. **History** — most-recent-first list. Click a row to expand the inline label form.

#### Keyboard shortcuts

| Key | Action |
|---|---|
| <kbd>Enter</kbd> (in instruction) | Start episode |
| <kbd>Esc</kbd> (anywhere) | Stop current episode |
| <kbd>Shift</kbd>+<kbd>Enter</kbd> (in instruction) | Newline |

The launching terminal also accepts <kbd>Esc</kbd> to stop the current episode and <kbd>Ctrl+C</kbd> to quit.

#### Editing min-z mid-session

The safety z floor can be updated between episodes directly in the Teleop card — the UI `POST`s to `/min_z` and the new value takes effect on the next episode. Updates are rejected while an episode is running (HTTP 409).

#### Label form

Every history row expands to a form for:

- **Experiment / Task** — edit after the fact; writes back to `meta.json`.
- **Valid** (Y/N) — was the episode usable?
- **Success** (Y/N) — did the demonstration succeed?
- **Score** — optional numeric.
- **Notes** — freeform string.

Hit **Save** to persist; the UI rewrites `{episode_dir}/meta.json` in place. Labels can be edited any number of times after the episode completes.

#### Persisted fields

The browser saves `experiment_service` / `teleop_service` form state to `localStorage` (key `teleop_service_state_v1`): experiment name, task name, max walltime. These survive page refreshes; clearing site data resets them.

#### CLI Reference (`run_teleop_ui.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--web-port` | `54325` | Web server port |
| `--port` | `/dev/ttyACM0` | SO-101 serial port |
| `--franky-service-url` | from `constants.py` | Franky service URL (URDF fetch) |
| `--no-gripper` | off | Skip gripper initialization |
| `--rate-hz` | `100.0` | Control loop rate (Hz) |
| `--min-z` | `0.23` | Initial safety z floor (editable in UI) |
| `--record-rate-hz` | `15.0` | Recording rate (Hz) |
| `--record-jpeg-quality` | from `constants.py` | JPEG quality for recorded images |
| `--output-dir` | `output` | Base output directory |
| `--task` | "" | Task name used in the run directory |
| `--dry-run` | off | Read SO-101 but don't command the robot or gripper |

### Recording

Records by default at 15 Hz (configurable via `--record-rate-hz`) while the control loop runs at 100 Hz. Each recorded step captures:

- **All 3 cameras** (left, wrist, right)
- **State**: measured joint positions/velocities + gripper position
- **Action**: commanded joint positions + gripper command

Data format is identical to the experiment runners — see [Experiment Logging](experiment-logging.md).

## Joint Mapping: SO-101 → Franka

The 5-DoF SO-101 maps to a subset of the 7-DoF Franka. Joints j2 and j4 stay at home (0.0).

Franka home position:
```
[0.0, -0.40, 0.0, -1.9, 0.0, 1.5, 0.0]
 j0    j1    j2    j3   j4   j5   j6
```

| SO-101 Joint | Clip Range | Franka Joint | Mapping |
|---|---|---|---|
| Shoulder Pan (0) | ±45° | j0 | `-so101[0]` |
| Shoulder Lift (1) | -60° to +90° | j1 | `so101[1]` |
| Elbow Flex (2) | -80° to +80° | j3 | `-π/2 - so101[2]` |
| Wrist Flex (3) | unclamped | j5 | `-so101[3] + π` |
| Wrist Roll (4) | unclamped | j6 | `-so101[4]` |

### Gripper Mapping

SO-101 gripper angle (0–90°) maps linearly to Robotiq position (0–255 bits):

| SO-101 | Robotiq | State |
|---|---|---|
| 0° | 255 | Closed |
| 90° | 0 | Open |

Commands sent only when position changes by >2 bits.

## Safety

### Table Collision Avoidance

FK (Pinocchio + URDF) checks that commanded configurations keep EE above `--min-z` (default 0.23m):

1. **Partial revert**: joints 1 and 3 reverted to last safe values
2. **Full revert**: entire configuration reverts if partial fix insufficient

### Joint Clipping

SO-101 values clipped to conservative ranges before mapping (see table above).

## Troubleshooting

| Problem | Fix |
|---|---|
| No `/dev/ttyACM*` | Check USB connection, try different port |
| Permission denied | `sudo chmod 666 /dev/ttyACM0` or add user to `dialout` group |
| Gripper not responding | Check 24V adapter, check `make gripper_service`, or use `--no-gripper` |
| Robot doesn't move | Verify FCI activated in Franka desk, franky_service running |
| `[safety] EE z=... reverting` | Move leader arm up, or adjust `--min-z` |

### 9.19 development
change relativefrankadynamic to make the joint streaming faster to reduce the control lag, the mapping for the 3rd, 4th joint on leader arm doesn't match the robot joint very well.