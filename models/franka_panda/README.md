# Panda arm kinematics

`panda.urdf` is derived from the official Franka description:

- Repository: https://github.com/frankarobotics/franka_ros
- Commit: `ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98`
- Input: `franka_description/robots/panda/panda.urdf.xacro`
- Generation: Xacro 2.1.1, `hand:=false gazebo:=false` (other arguments default).
- License and upstream attribution: adjacent `LICENSE` and `NOTICE`.

During generation, `$(find franka_description)` was resolved to the downloaded
package directory. After expansion, all `visual` and `collision` elements were
removed so no external mesh files or ROS installation are needed. Joint origins,
axes, limits, and inertial data are retained. This is a kinematic model, not a
rendering or collision model.

The model has seven revolute joints and a fixed `panda_link8` flange frame. It does
not model an attached Franka Hand, Robotiq gripper, or tool TCP. The teleoperation
height floor therefore applies to the flange, not the lowest point of your tool.

From the DroidPlus directory:

```bash
export FRANKA_URDF_PATH="$PWD/models/franka_panda/panda.urdf"
.venv/bin/python scripts/run_teleop_cli.py \
  --controller spacemouse --dry-run --no-record --gripper none
```

This explicitly selects the Panda model and bypasses the robot service's `/urdf`
endpoint. Set the variable again in each new shell that runs teleoperation.
Generation and validation did not connect to or command a physical robot.
