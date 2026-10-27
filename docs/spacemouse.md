# SpaceMouse teleoperation

Control the Franka with a USB SpaceMouse instead of the SO-101 leader arm.
Motion is restricted to XYZ translation and yaw about the robot base Z axis.
Roll and pitch input are ignored; the controller maintains the starting tilt.
Both existing entry points accept `--controller spacemouse`; SO-101 remains
the default. Recordings retain the existing joint action/state format, cameras,
episode labels, and phase annotations. Metadata identifies `teleop_spacemouse`
and stores the controller settings.

From the repository directory, install in your robot workstation environment:

```bash
pip install -e '.[spacemouse,analysis]'
python scripts/run_teleop_cli.py --controller spacemouse --dry-run --no-record --gripper none
python scripts/run_teleop_ui.py --controller spacemouse --dry-run --gripper none
```

The SpaceMouse plugs into the computer running the Python entry point, including
when using the web UI from another computer. Select a model explicitly with
`--spacemouse-device 'SpaceMouse Compact'`, or leave it unset for auto-detection.
The SpaceMouse extra does not require LeRobot.

Use the [PySpaceMouse installation instructions](https://spacemouse.kubaandrysek.cz/)
for Linux HID permissions and native HID library setup. This integration uses
PySpaceMouse 2.1's device API and explicit `HID_Z_UP` axis convention. Driver
requirements can differ from newer unreleased upstream code.

## Panda service returns `/urdf` 404

For a Franka Panda, a local arm model is provided at
[`models/franka_panda/panda.urdf`](../models/franka_panda/panda.urdf).
From the DroidPlus directory, set this before running either entry point:

```bash
export FRANKA_URDF_PATH="$PWD/models/franka_panda/panda.urdf"
```

This bypasses the HTTP model lookup. The model ends at the Panda flange
(`panda_link8`); it does not include gripper/tool offsets. The height floor applies
to that flange. See [model provenance](../models/franka_panda/README.md).
Do not use this Panda model for an FR3.

## Controls

1. Start an episode using the usual CLI or web UI controls.
2. Center the cap and release both buttons to arm. If necessary, move and release
   the cap once to produce an initial HID report.
3. Hold button 0 to enable motion. Push/pull the cap for translation; twist for yaw. Tilting the cap does not
   command roll or pitch. Release button 0 to hold the last commanded pose.
4. While holding button 0, press button 1 to toggle the gripper. The first press
   requests close, the next requests open. Button indices are zero-based and
   configurable with `--enable-button` and `--gripper-button`.
5. Stop the episode with the existing ESC or web UI stop control.

Motion is expressed in the **robot base frame**, using the EE frame selected
from the existing URDF loader (printed at startup, commonly `panda_link8`).
The default maps driver x/y/z/yaw to base x/y/z/rz. After axis remapping,
roll and pitch are suppressed. Inspect axis
orientation on your setup before live operation; `--axis-map=-2,1,3,-5,4,6`
shows how to swap axes and invert signs. Values must form a signed permutation
of 1 through 6.

| Option | Default | Meaning |
| --- | --- | --- |
| `--linear-speed` | 0.05 | Maximum translation vector norm, m/s |
| `--angular-speed` | 0.3 | Maximum rotation vector norm, rad/s |
| `--joint-speed` | 0.3 | Maximum commanded speed per joint, rad/s |
| `--deadzone` | 0.1 | Normalized cap input dead zone |
| `--input-timeout` | 0.25 | Maximum active report age, seconds |
| `--min-z` | 0.23 | Minimum end-effector height in base frame, metres |

Differential IK starts at measured joints on each episode. It uses a damped
Jacobian inverse and rejects targets outside URDF joint bounds (with a 0.01 rad
margin) or below the height floor. Targets never accumulate rejected motion.
A measured-to-commanded error over 0.15 rad ends the episode. Integration uses monotonic elapsed time. Each step is capped at 50 ms,
and a control interval exceeding 250 ms ends the episode rather than catching up. Missing URDF/Pinocchio prevents SpaceMouse operation.

Active input older than the timeout or HID read errors end the episode and invoke
the existing robot stop method. A neutral idle device may legitimately send no
reports, so neutral inactivity alone does not end the episode. Holding enable
without moving can time out on devices that only report changes; release and
restart the episode. Software stopping depends on the host and robot connection;
the height check is not full arm/environment collision detection.

Dry-run still reads the robot's initial joints and uses existing service/camera
connections, but sends no arm, gripper, or startup stop commands. It integrates
virtual targets for checking mappings and recording; it is not a physics simulator.
Remove `--dry-run` for live control after validating the setup. No real robot or
SpaceMouse was used for the automated tests.

## Independent recording and diagnostics

For SpaceMouse episodes, camera acquisition and recording run in a separate
worker. The control loop publishes one latest snapshot containing measured
joints, the accepted joint target, gripper command, phase and wall timestamp.
The worker samples the latest snapshot before requesting images. Slow cameras
reduce recording throughput without blocking motion or building a stale queue.
Images are fetched sequentially; the timestamp identifies the control snapshot,
not a hardware-synchronized camera exposure. Gripper observations are read by
the recording worker. This preserves the existing eight-element state/action
schema (seven joints plus gripper), with zero action velocities as before.

Robot stop is requested before joining the recording worker, and finalization
waits for any in-flight frame. Worker errors stop the episode. Metadata reports
measured recording rate for these episodes rather than nominal tick decimation.

Once per second, `[spacemouse]` prints the arm/hold/rejection status, mapped XYZ
and yaw input, enable state, and measured control/recording rates. A height or
joint-limit rejection is explicit. NUC code and its joint-target API are unchanged.

## Optional Deoxys-style buttons

The default `--button-mode clutch` still requires holding button 0 to move and
uses button 1 to toggle the gripper. To use free motion after arming, left-button
hold-to-close/release-to-open, and right-button stop:

```bash
.venv/bin/python scripts/run_teleop_cli.py \
  --controller spacemouse --gripper franka --task spacemouse_teleop \
  --button-mode deoxys --gripper-button 0 --enable-button 1
```

Center the cap and release buttons at episode start. In this mode you do not
hold an enable button during motion. The right button ends the episode; it does
not move the robot to a reset pose. Button indices can be swapped if your device
reports them differently. This changes input behavior only, not the NUC backend.

## Isolated runner and latency debugging

SpaceMouse episodes execute in `droid_plus/datagen/spacemouse_runner.py`.
`teleop_runner.py` retains the original SO-101 episode loop; it does not import
or branch on SpaceMouse. Both existing CLI/UI entry points select the appropriate
runner. Session configuration and recording finalization remain shared.

A dedicated CLI is also available, with the same options except `--controller`:

```bash
export FRANKA_URDF_PATH="$PWD/models/franka_panda/panda.urdf"
.venv/bin/python scripts/run_spacemouse_cli.py \
  --gripper franka --task spacemouse_teleop --rate-hz 50
```

For read-only timing investigation, add `--dry-run --no-record --gripper none`.
Dry-run measures state reads and IK but does not measure live target-post latency.
Periodic diagnostics show `state_get_ms`, `input_ms`, `ik_ms`, `target_post_ms`,
`gripper_ms`, and `interval_ms`. On failure the current and previous tick timings
are printed before cleanup. These distinguish robot-service/HTTP delays from
input or IK delays. The 250 ms control-stall stop is unchanged; separation alone
does not resolve a slow connection or robot service.
