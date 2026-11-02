"""Offline checks for the six-phase drawer/cube collection profile."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

from droid_plus.datagen.teleop_runner import TeleopSessionConfig, finalize_teleop_episode_recording
from droid_plus.eval.episode_runner import EpisodeConfig, EpisodeResult
from droid_plus.logging import EpisodeRecorder
from droid_plus.logging.phase_labels import PhaseSelector
from scripts import run_teleop_ui as ui
from scripts.run_teleop_drawer_6phase_ui import PHASE_NAMES


class DrawerSixPhaseTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.multiple(ui, PHASE_NAMES=PHASE_NAMES, SAVE_PHASE_PROMPTS=True,
                                 REQUIRED_CAMERAS=("left", "wrist"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ui_accepts_six_phases_and_renders_exact_names(self):
        state = SimpleNamespace(status="running", lock=threading.Lock(), phases=PhaseSelector())
        with patch.object(ui, "_state", state):
            for phase in range(6):
                self.assertEqual(ui.episode_phase({"phase_index": phase})["phase_index"], phase)
            for phase in (6, -1, True, "5"):
                with self.assertRaises(HTTPException):
                    ui.episode_phase({"phase_index": phase})
        html = ui.root().body.decode()
        for name in PHASE_NAMES:
            self.assertIn(name, html)
        self.assertNotIn("__PHASE_NAMES_JSON__", html)
        self.assertIn("0–5", html)

    def test_camera_preflight_blocks_missing_or_invalid_images(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        droid = SimpleNamespace(left_camera_id="31262010", wrist_camera_id="14633207",
                                get_left_image=lambda: rgb, get_wrist_image=lambda: rgb)
        ui.verify_required_cameras(droid)
        droid.wrist_camera_id = ""
        with self.assertRaises(RuntimeError):
            ui.verify_required_cameras(droid)
        droid.wrist_camera_id = "14633207"
        droid.get_wrist_image = lambda: None
        with self.assertRaises(RuntimeError):
            ui.verify_required_cameras(droid)

    def test_prompts_images_and_metadata_saved_during_collection(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(directory, 0, phase_prompts=dict(enumerate(PHASE_NAMES)))
            labels = [phase for phase in range(6) for _ in range(8)]
            for i, phase in enumerate(labels):
                recorder.record_step(seq=i, left_rgb=rgb, wrist_rgb=rgb,
                                     state_positions=np.zeros(8), state_velocities=np.zeros(8),
                                     action_positions=np.zeros(8), action_velocities=np.zeros(8), phase_index=phase)
            result = EpisodeResult(episode_idx=0, seq=len(labels), t_start=1, t_end=10,
                                   inference_times=[], recorder=recorder,
                                   stopped_by_limit=False, stopped_by_caller=True)
            finalize_teleop_episode_recording(result=result, config=EpisodeConfig(),
                                              session=TeleopSessionConfig(rate_hz=10, record_rate_hz=10))
            episode = Path(recorder.episode_dir)
            rows = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len(labels))
            for row in rows:
                self.assertEqual(row["phase_prompt"], PHASE_NAMES[row["phase_index"]])
                self.assertEqual(list(row)[-2:], ["phase_index", "phase_prompt"])
                for camera in ("left", "wrist"):
                    self.assertTrue((episode / row["images"][camera]).is_file())
            metadata = json.loads((episode / "meta.json").read_text())
            self.assertFalse(metadata["phase_annotation"]["needs_review"])
            self.assertEqual(metadata["phase_annotation"]["expected_phase_sequence"], list(range(6)))
            self.assertEqual(metadata["phase_prompt_mapping"]["5"], "close_drawer_and_home")

    def test_incomplete_episode_is_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(directory, 0, phase_prompts=dict(enumerate(PHASE_NAMES)))
            for i in range(8):
                recorder.record_step(seq=i, state_positions=np.zeros(8), state_velocities=np.zeros(8),
                                     action_positions=np.zeros(8), action_velocities=np.zeros(8), phase_index=0)
            result = EpisodeResult(episode_idx=0, seq=8, t_start=1, t_end=2,
                                   inference_times=[], recorder=recorder,
                                   stopped_by_limit=False, stopped_by_caller=True)
            finalize_teleop_episode_recording(result=result, config=EpisodeConfig(),
                                              session=TeleopSessionConfig(rate_hz=10, record_rate_hz=10))
            metadata = json.loads((Path(recorder.episode_dir) / "meta.json").read_text())
            self.assertTrue(metadata["phase_annotation"]["needs_review"])
            self.assertTrue(any("Expected phases" in issue for issue in metadata["phase_annotation"]["issues"]))

    def test_spacemouse_ui_records_all_six_phase_prompts(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        droid = SimpleNamespace(left_camera_id='31262010', wrist_camera_id='14633207',
                                right_camera_id='', get_left_image=lambda: rgb,
                                get_wrist_image=lambda: rgb)

        def record_episode(**kwargs):
            recorder = kwargs['recorder']
            for phase in range(6):
                for frame in range(8):
                    recorder.record_step(
                        seq=phase * 8 + frame, left_rgb=rgb, wrist_rgb=rgb,
                        state_positions=np.zeros(8), state_velocities=np.zeros(8),
                        action_positions=np.zeros(8), action_velocities=np.zeros(8), phase_index=phase,
                    )
            result = EpisodeResult(0, 48, 1, 6, [], recorder, False, True)
            result.actual_record_rate_hz = 10.0
            return result

        with tempfile.TemporaryDirectory() as directory:
            state = ui.AppState(
                droid=droid, teleop=object(), gripper_initialized=False,
                pin_model=None, pin_data=None, ee_frame='tool',
                session=TeleopSessionConfig(policy_name='teleop_spacemouse',
                    controller_info={'controller': 'spacemouse', 'rotation': 'full'}),
                so101_port='SpaceMouse (auto)', base_run_dir=directory,
                current_config=EpisodeConfig(task='drawer', instruction='put the cube into the drawer'),
                status='running',
            )
            with patch.object(ui, 'run_spacemouse_episode', side_effect=record_episode) as mouse_runner, \
                 patch.object(ui, 'run_teleop_episode') as arm_runner, \
                 patch.object(ui, 'compute_and_save_ee_trajectory_single'):
                ui._run_episode_thread(state)
            mouse_runner.assert_called_once()
            arm_runner.assert_not_called()
            self.assertEqual(state.status, 'idle')
            self.assertEqual(len(state.history), 1)
            episode = Path(directory) / 'episode_000'
            meta = json.loads((episode / 'meta.json').read_text())
            self.assertEqual(meta['policy']['name'], 'teleop_spacemouse')
            self.assertEqual(meta['control']['motion_mode'], 'xyz_rpy')
            self.assertEqual(meta['recording_status'], 'complete')
            self.assertFalse(meta['phase_annotation']['needs_review'])
            rows = [json.loads(line) for line in (episode / 'steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 48)
            for row in rows:
                self.assertEqual(row['phase_prompt'], PHASE_NAMES[row['phase_index']])


if __name__ == "__main__":
    unittest.main()
