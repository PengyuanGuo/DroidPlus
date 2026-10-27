<p align="center">
  <img src="docs/images/droid_plus_logo.png" alt="DROID+ logo" width="220">
</p>

# DROID+

DROID+ is an improved hardware control stack for [DROID](https://droid-dataset.github.io), which contains robot control, camera, and gripper services and clients for the DROID setup.
DROID+ runs policies using joint position.

## Documentation

- [Hardware Setup](docs/hardware-setup.md) - DROID hardware components and setup
- [Software Setup](docs/software-setup.md) - NUC flashing, workstation setup, camera serial numbers
- [Adding New Experiments](docs/adding-experiments.md) - `experiments.json` schema for defining preset experiments
- [Policy Evaluation](docs/eval.md) - CLI + Web UI workflow for running closed-loop policy episodes
- [Teleoperation](docs/teleop.md) - CLI + Web UI workflow for SO-101 leader-arm teleop
- [Experiment Logging](docs/experiment-logging.md) - Data recording format, post-episode prompts, and analysis

## Installation

```bash
git clone https://github.com/NVlabs/DroidPlus.git
cd DroidPlus

python3 -m venv venv
source venv/bin/activate

pip install -e .

# Optional extras:
pip install -e ".[all]"        # Compatible non-teleop extras
pip install -e ".[gripper]"    # Gripper control
pip install -e ".[policy]"     # Policy clients
pip install -e ".[teleop]"     # SO-101 teleoperation
pip install -e ".[analysis]"   # Analysis tools (matplotlib, pinocchio)
pip install -e ".[dev]"        # Development (pytest, ruff, mypy)
```

Install `policy` and `teleop` in separate environments: LeRobot 0.4.x requires
NumPy 2, while `openpi-client` currently requires NumPy 1.x. The `all` extra
therefore excludes `teleop`.

This project will download and install additional third-party open source
software projects. Review the license terms of these open source projects
before use.

### Franka Robot-Control Dependency

DROID+ does not distribute, bundle, or automatically install
[`franky-control`](https://github.com/TimSchneider42/franky). The
`franky-service` entry point requires a separately obtained installation.

The upstream `LICENSE` identifies `franky-control` as LGPL-3.0-or-later, while
its README contains additional wording concerning commercial use. Review and
comply with the upstream terms before installing it. NVIDIA does not grant any
rights to `franky-control`.

After confirming that the upstream terms are appropriate for your use:

```bash
pip install franky-control==1.1.3
```

Optional video export tools:

```bash
sudo apt install ffmpeg
```

`ffmpeg` is a user-installed system tool. It is not installed by `pip install`
and is not distributed with DROID+.

### ZED Camera Setup (Workstation only)

```bash
make setup_zed
```

## Bring-up Procedure

### Step 1: NUC (Franka control host)

1. Power on the NUC. In GRUB, select the Ubuntu entry with `RT` in the name (real-time kernel).
2. `source venv/bin/activate`
3. Confirm that the separately obtained `franky-control` installation is available.
4. `make franky_service` — serves the Franka control API on port 54321.
5. In a browser, open `https://<franka-robot-ip>/desk` (the Franka desk interface):
   - If "self tests are overdue", press **Acknowledge & Execute** and wait.
   - Open **Joints** and unlock.
   - Click your robot name (top-left) → **Activate FCI**.

### Step 2: Workstation (camera + gripper services)

In two separate terminals:

1. `source venv/bin/activate && make camera_service` — ZED camera service on port 54322.
2. `source venv/bin/activate && make gripper_service` — Robotiq 2F-85 gripper service on port 54323.

### Step 3: Verify

```bash
make home          # Move robot to home position
make open          # Open gripper
```

If both succeed, the system is ready to run experiments.

## Running Experiments

Four entry points share the same recording format (see [Experiment Logging](docs/experiment-logging.md)):

```bash
# CLI — closed-loop policy episodes
python scripts/run_experiment_cli.py --exp banana_in_bowl
python scripts/run_experiment_cli.py --instruction "pick up the can"

# Web UI — browser interface for policy episodes
python scripts/run_experiment_ui.py

# Teleop CLI — SO-101 leader arm, records all cameras
python scripts/run_teleop_cli.py

# Teleop Web UI — browser interface for SO-101 teleop
python scripts/run_teleop_ui.py
```

The CLI entry points use SPACE to start / ESC to stop episodes, prompt for labels (valid/success/score/notes), and save to `output/` (override with `--output-dir`). The web UIs use Enter / Esc and collect labels inline — see [Policy Evaluation](docs/eval.md) and [Teleoperation](docs/teleop.md) for the workflow details.

Make targets for common configurations:

```bash
make run_experiment_900 INSTRUCTION="pick up the can" NOTES="experiment notes"
```

## Teleoperation

[SpaceMouse teleoperation](docs/spacemouse.md) is available with `--controller spacemouse` in both the CLI and web UI.

See [docs/teleop.md](docs/teleop.md) for full details including joint mapping and safety features.

```bash
python scripts/run_teleop_cli.py
python scripts/run_teleop_cli.py --port /dev/ttyACM1 --no-gripper
python scripts/run_teleop_cli.py --no-record
python scripts/run_teleop_ui.py                    # browser UI
```

Requires the `teleop` extra, which installs LeRobot 0.4.x with Feetech support,
and the `analysis` extra for FK safety checks:

```bash
pip install -e ".[teleop,analysis]"
```

## Common Commands

```bash
make home          # Move robot to home position
make stop          # Stop robot
make reset         # Stop, go home, open gripper
make open          # Open gripper
make close         # Close gripper
make drop          # Drop object (open gripper)
```

## Python API

```python
from droid_plus import DroidPlus, FrankyClient, CameraClient, GripperClient

droid = DroidPlus()
droid.move_to_home()
left_image = droid.get_left_image()
joint_state = droid.get_current_joint_state()
```

## Data Export

```bash
make export_parquet  RUN_DIR=output/experiment_20260118_123456
make export_lerobot  RUN_DIR=output/experiment_20260118_123456            # LeRobot v2.1 local export
make export_lerobot  RUN_DIR=output/experiment_20260118_123456 PUSH=hugo/my_dataset PRIVATE=1
make video           RUN_DIR=output/experiment_20260118_123456 FPS=10
```

`make export_lerobot` writes a local LeRobot v2.1 dataset. See
[docs/experiment-logging.md](docs/experiment-logging.md) for details.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUC_IP` | `localhost` | Host running `franky_service` |
| `FRANKY_SERVICE_URL` | `http://{NUC_IP}:54321` | Franky service URL |
| `CAMERA_SERVICE_URL` | `http://127.0.0.1:54322` | Camera service URL |
| `GRIPPER_SERVICE_URL` | `http://127.0.0.1:54323` | Gripper service URL |
| `FRANKY_ROBOT_IP` | `localhost` | Franka controller IP/hostname used by `franky_service` |
| `WRIST_CAMERA_SERIAL` | unset | ZED wrist camera serial |
| `LEFT_CAMERA_SERIAL` | unset | ZED left camera serial |
| `RIGHT_CAMERA_SERIAL` | unset | ZED right camera serial |
| `POLICY_HOST` | `127.0.0.1` | Policy server host |
| `POLICY_PORT` | `8000` | Policy server port |

## License

DROID+ is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE)
for the full license text.

## Contributions

This project is currently not accepting contributions.

## Contributors

DROID+ is built by [Seattle Robotics Lab](https://research.nvidia.com/labs/srl/): Hugo Hadfield, Xuning Yang, Rowland O'Flaherty, Lars Johannsmeier.
