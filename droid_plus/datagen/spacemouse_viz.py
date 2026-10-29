"""Viser view of SpaceMouse teleop: measured arm, commanded target, input and floor.

Kinematic skeleton only (the local URDF has no meshes). The hand drawn below the
flange is an approximate Franka Hand for orientation and gripper state; the
height floor (``min_z``) applies to the flange, not the fingertips.
"""
from __future__ import annotations

import time

import numpy as np

_TARGET = (255, 140, 0)
_MEASURED = (150, 150, 150)
_HAND_TCP = 0.1034  # Franka Hand flange-to-fingertip-centre offset, metres.
_HAND_YAW = -np.pi / 4  # Franka Hand is mounted rotated -45 deg about the flange z.
_TRAIL = 3000


class TeleopVisualizer:
    """Updated from the control loop; rate-limited and safe to call every tick."""
    def __init__(self, model, data, ee_frame, min_z, *, port=8080, max_rate_hz=30.0):
        import pinocchio as pin
        import viser

        self.pin, self.model, self.data = pin, model, data
        self.frame = model.getFrameId(ee_frame)
        self.min_z = min_z
        self.period = 1.0 / max_rate_hz
        self.last_update = 0.0
        self.trail = []
        self.server = viser.ViserServer(port=port)
        scene, gui = self.server.scene, self.server.gui
        scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1, section_size=0.5)
        scene.add_frame("/base", axes_length=0.15, axes_radius=0.006)
        scene.add_box("/floor", color=(220, 60, 60), dimensions=(1.2, 1.2, 0.002), opacity=0.15,
                      position=(0.5, 0.0, min_z))
        scene.add_label("/floor/label", f"flange floor z={min_z:.2f} m", position=(0.95, -0.55, 0.0))
        self.measured = scene.add_line_segments("/measured", np.zeros((1, 2, 3)), _MEASURED, thickness=0.02)
        self.target = scene.add_line_segments("/target", np.zeros((1, 2, 3)), _TARGET, thickness=0.012)
        self.hand = scene.add_line_segments("/target_hand", np.zeros((1, 2, 3)), _TARGET, thickness=0.008)
        self.ee = scene.add_frame("/target_ee", axes_length=0.08, axes_radius=0.004)
        self.arrow = scene.add_line_segments("/input", np.zeros((1, 2, 3)), (30, 120, 255), thickness=0.01)
        self.trail_cloud = scene.add_point_cloud("/trail", np.zeros((1, 3)), (255, 140, 0), point_size=0.004)
        with gui.add_folder("SpaceMouse teleop"):
            self.status = gui.add_text("Status", "starting", disabled=True)
            self.input = gui.add_text("Input", "", disabled=True)
            self.gripper = gui.add_text("Gripper", "", disabled=True)
            self.pose = gui.add_text("Target EE", "", disabled=True)
            self.height = gui.add_text("Height", "", disabled=True)
            self.tracking = gui.add_text("Tracking", "", disabled=True)
            gui.add_markdown("Grey: measured arm. Orange: commanded target. Blue: SpaceMouse input.")
            gui.add_button("Clear trail").on_click(lambda _: self.reset())
        print(f"Viser visualizer at http://localhost:{port}")

    def _chain(self, q):
        full = self.pin.neutral(self.model)
        full[:7] = q
        self.pin.forwardKinematics(self.model, self.data, full)
        self.pin.updateFramePlacement(self.model, self.data, self.frame)
        points = [np.zeros(3)] + [self.data.oMi[j].translation.copy() for j in range(1, 8)]
        flange = self.data.oMf[self.frame]
        points.append(flange.translation.copy())
        points = np.array(points)
        return np.stack([points[:-1], points[1:]], axis=1), flange.translation.copy(), flange.rotation.copy()

    def reset(self):
        self.trail = []

    def update(self, *, target, twist, enabled, gripper_closed, status, measured=None):
        now = time.monotonic()
        if now - self.last_update < self.period:
            return
        self.last_update = now
        segments, flange, rotation = self._chain(np.asarray(target, dtype=float))
        self.target.points = segments
        hand_rot = rotation @ self.pin.exp3(np.array([0.0, 0.0, _HAND_YAW]))
        tcp = flange + rotation[:, 2] * _HAND_TCP
        half = 0.0 if gripper_closed else 0.04
        palm = flange + rotation[:, 2] * 0.06
        fingers = [palm + hand_rot[:, 1] * s * max(half, 0.008) for s in (-1, 1)]
        self.hand.points = np.array([
            [flange, palm], [fingers[0], fingers[1]],
            [fingers[0], fingers[0] + (tcp - palm)], [fingers[1], fingers[1] + (tcp - palm)],
        ])
        self.ee.position = flange
        self.ee.wxyz = self.pin.Quaternion(rotation).coeffs()[[3, 0, 1, 2]]
        twist = np.asarray(twist, dtype=float)
        direction = twist[:3] / max(np.linalg.norm(twist[:3]), 1e-9) if np.any(twist[:3]) else np.zeros(3)
        self.arrow.points = np.array([[tcp, tcp + direction * 0.15 * (1.0 if enabled else 0.4)]])
        self.trail.append(tcp)
        self.trail = self.trail[-_TRAIL:]
        self.trail_cloud.points = np.array(self.trail)
        self.trail_cloud.colors = np.tile(np.array(_TARGET, dtype=np.uint8), (len(self.trail), 1))

        error = ""
        if measured is not None:
            measured = np.asarray(measured, dtype=float)
            self.measured.points = self._chain(measured)[0]
            error = f"max joint error {np.max(np.abs(measured - target)):.3f} rad"
        self.measured.visible = measured is not None
        self.status.value = status
        self.input.value = (f"enable={'ON' if enabled else 'off'}  v=[{' '.join(f'{v:+.3f}' for v in twist[:3])}] m/s"
                            f"  yaw={twist[5]:+.2f} rad/s")
        self.gripper.value = "CLOSED" if gripper_closed else "open"
        self.pose.value = f"x={flange[0]:+.3f} y={flange[1]:+.3f} z={flange[2]:+.3f} m"
        self.height.value = (f"flange {flange[2] - self.min_z:+.3f} m above floor;"
                             f" fingertips ~{tcp[2]:.3f} m")
        self.tracking.value = error or "dry run / sim: no measured motion"

    def close(self):
        self.server.stop()
