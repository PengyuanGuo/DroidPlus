# Changelog

All notable changes to DROID+ are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- SpaceMouse teleoperation (`docs/spacemouse.md`): XYZ + roll/pitch/yaw differential IK
  (`--rotation full|yaw`, separate rotation dead zone), 10 cm/s default speed, viser view
  (`--viser`), virtual-arm simulator (`scripts/spacemouse_viser.py`), and input recorder with
  guided axis check and offline replay (`scripts/record_spacemouse.py`).
- SpaceMouse task launchers with per-frame phase names: MARS crossword (4 phases), drawer/cube
  (6 phases), and drawer chicken-leg search (6 phases, `commands/spacemouse_drawer_chicken_6phase`).
- Branching phase profiles: `phase_transitions`/`final_phases` review episodes whose phase order
  depends on the scene; recorded in `meta.json`.
- `franky_service` `/health`: `control_thread_alive`, `control_heartbeat_age_s`, target/control
  timing; optional all-thread stack dumps (`SIGUSR1`, `FRANKY_TRACEBACK_INTERVAL_S`).

### Changed

- Shared teleop height floor `DEFAULT_MIN_EE_Z` 0.23 -> 0.12 m (flange; Hand fingertips ~0.10 m lower).
- `HOME_POSITION` joint 7 is now π/4 (was 0).
- Recording worker errors name the failing stage (camera, gripper observation, ...).
- SpaceMouse targets are posted from a background thread; control never waits on HTTP.

### Fixed

- `franky_service`: serialize robot calls and cache joint state to avoid the franky
  `move()`/state deadlock; skip redundant target preemptions.
- `franky_service`: cache the Franka Hand width instead of polling it (each read stalled the
  whole service for 30–100 ms); no repeated close commands on a grasped object.
- `franky_service`: control thread no longer dies when error recovery fails at startup.
- SpaceMouse: holding a button with a centered cap no longer ends the episode; camera wait
  only covers cameras the run uses.

## [0.1.0] - 2026-08-19

### Added

- Initial public release of DROID+: Franka robot control, camera, and gripper
  services and clients; policy evaluation (CLI + web UI); SO-101 leader-arm
  teleoperation; episode recording with parquet and LeRobot v2.1 export; and
  end-effector trajectory analysis tools.
- Project logo in the README.
- This changelog.

[0.1.0]: https://github.com/NVlabs/DroidPlus/releases/tag/v0.1.0
