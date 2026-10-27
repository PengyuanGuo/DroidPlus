# Six-phase drawer and cube collection

Run the existing camera and Franky services, then launch this task-specific UI:

```bash
cd /home/atomic/Project/DroidPlus
source env_variables.md
.venv-teleop/bin/python scripts/run_teleop_drawer_6phase_ui.py \
  --gripper franka \
  --franky-service-url http://192.168.9.167:54321 \
  --left-camera-id 31262010 \
  --wrist-camera-id 14633207 \
  --port /dev/ttyACM0 \
  --web-port 54325 \
  --output-dir output_1024_drawer_6phase
```

Open http://localhost:54325. Enter the task instruction and start with Enter.
Use browser buttons or digit keys outside text fields, or the teleop terminal:

| Key / phase_index | phase_prompt |
|---|---|
| 0 | grasp_handle_open |
| 1 | open_drawer_and_home |
| 2 | pick_cube_and_lift |
| 3 | carry_place_and_home |
| 4 | grasp_handle_close |
| 5 | close_drawer_and_home |

Each episode starts at phase 0. Demonstrate the six stages in order, then press
Esc to stop and enter the episode's validity/success labels. Start a new episode
for the next full demonstration. A phase change only changes the recorded label;
it does not execute a skill, stop the arm, or move it home. Perform each movement
with the SO-101 leader, including the `and_home` parts.

This launcher reuses `run_teleop_ui.py`, DroidPlus clients, and the existing
collection loop. It defaults to the Franka Hand and the two camera IDs shown
above; explicit command-line values override these defaults. No NUC changes are
needed. Existing teleop launchers retain their previous phase behavior.

Left and wrist IDs must be nonempty and both cameras must return RGB images
before startup and before each recording. The launcher stops that startup or
episode attempt if these checks fail. Network failures during a recording are
reported by the existing recording loop; inspect the saved episode afterward.

Every recorded step has images and the phase fields at the end, for example:

```json
{"images":{"left":"left/000000000.jpg","wrist":"wrist/000000000.jpg"},"phase_index":2,"phase_prompt":"pick_cube_and_lift"}
```

The real rows also include timestamps, state and action values. Prompts are saved
during collection and do not need to be added afterward. `meta.json` includes the
phase-name mapping and review flags for missing, out-of-order, or short phases;
review warnings do not discard data or automatically change labels.

Start with a short test episode and verify JPEGs in `left/` and `wrist/` and
nonempty image paths in `steps.jsonl`. The UI uses the existing gripper homing and
teleoperation startup behavior. `--dry-run` skips arm/gripper commands but still
requires the leader connection, camera observations, and robot state service.
