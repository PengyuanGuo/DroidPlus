"""Phase input, recording, and export tests that run without robot hardware."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow.parquet as pq

from droid_plus.datagen.teleop_runner import (
    TeleopSessionConfig,
    finalize_teleop_episode_recording,
    run_teleop_episode,
)
from droid_plus.eval.episode_runner import EpisodeConfig
from droid_plus.logging import EpisodeRecorder
from droid_plus.logging.export_parquet import export_episode
from droid_plus.logging.lerobot_export import (
    ExportConfig,
    LeRobotExporter,
    _build_features,
    _load_episode,
    _write_episode_parquet,
)
from droid_plus.logging.phase_labels import PhaseSelector, review_phase_labels
from scripts.run_teleop_cli import _make_cli_should_stop


class PhaseCollectionTests(unittest.TestCase):
    def test_cli_keys_and_reset(self):
        phases = PhaseSelector()
        keys = Mock()
        keys.poll_char.side_effect = ["1", "1", "2", "\x1b"]
        with patch("scripts.run_teleop_cli.sys.stdin.isatty", return_value=True):
            stop = _make_cli_should_stop(keys, [False], phases)
            for expected in [1, 1, 2]:
                self.assertFalse(stop())
                self.assertEqual(phases.get(), expected)
            self.assertTrue(stop())
        self.assertEqual(PhaseSelector().get(), 0)
        for invalid in [True, -1, 128, 1.5, "1", None]:
            with self.assertRaises(ValueError):
                phases.set(invalid)

    def test_review(self):
        self.assertFalse(review_phase_labels([0, 0, 1, 1], min_frames=2)["needs_review"])
        for labels in [[], [1, 1], [0, 2], [0, 1, 0], [0, 0, 1]]:
            self.assertTrue(review_phase_labels(labels, min_frames=2)["needs_review"])

    def test_record_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, 0)
            for i, phase in enumerate([0, 0, 1, 1, 2]):
                recorder.record_step(
                    seq=i, left_rgb=np.zeros((16, 16, 3), dtype=np.uint8), state_positions=np.zeros(8), state_velocities=np.zeros(8),
                    action_positions=np.ones(8), action_velocities=np.zeros(8),
                    phase_index=phase,
                )
            recorder.finalize(task="test task", control_info={"record_rate_hz": 15})
            episode_dir = Path(recorder.episode_dir)
            rows = [json.loads(line) for line in (episode_dir / "steps.jsonl").read_text().splitlines()]
            self.assertEqual([row["phase_index"] for row in rows], [0, 0, 1, 1, 2])
            exported = LeRobotExporter(ExportConfig(
                run_dir=episode_dir, out_dir=Path(tmp) / "lerobot",
            )).export()
            exported_table = pq.read_table(exported / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(exported_table["phase_index"].to_pylist(), [0, 0, 1, 1, 2])
            info = json.loads((exported / "meta/info.json").read_text())
            self.assertEqual(info["features"]["phase_index"]["dtype"], "int64")
            stats = json.loads((exported / "meta/stats.json").read_text())
            self.assertEqual(stats["phase_index"]["max"], [2.0])
            ep = _load_episode(episode_dir)
            out = Path(tmp) / "export.parquet"
            _write_episode_parquet(out, episode=ep, episode_index=0, global_index_start=0, task_index=0, fps=15)
            table = pq.read_table(out)
            self.assertEqual(table["phase_index"].to_pylist(), [0, 0, 1, 1, 2])
            self.assertEqual(str(table.schema.field("phase_index").type), "int64")
            self.assertEqual(_build_features(224, 224, 15)["phase_index"]["dtype"], "int64")
            self.assertEqual(pq.read_table(export_episode(episode_dir))["phase_index"].to_pylist(), [0, 0, 1, 1, 2])

            # Missing labels on old episodes stay explicitly unlabelled.
            for row in rows:
                row.pop("phase_index")
            steps = episode_dir / "steps.jsonl"
            steps.write_text("\n".join(json.dumps(row) for row in rows))
            ep = _load_episode(episode_dir)
            _write_episode_parquet(out, episode=ep, episode_index=0, global_index_start=0, task_index=0, fps=15)
            self.assertEqual(pq.read_table(out)["phase_index"].to_pylist(), [-1] * 5)
            rows[0]["phase_index"] = 0
            steps.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                _load_episode(episode_dir)

    def test_runner_uses_phase_at_frame_start(self):
        phases = PhaseSelector()
        droid = Mock(left_camera_id="left", wrist_camera_id="", right_camera_id="")
        droid.get_current_joint_state.return_value = {"positions": [0.0] * 7}
        # Simulate a keypress during camera IO. It must apply to the next frame.
        def image(**kwargs):
            phases.set(1)
            return np.zeros((4, 4, 3), dtype=np.uint8)
        droid.get_left_image.side_effect = image
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, 0)
            config = EpisodeConfig(action_step_limit=2)
            session = TeleopSessionConfig(rate_hz=100, record_rate_hz=100, dry_run=True)
            with patch("droid_plus.datagen.teleop_runner.action_to_so101_joints_deg", return_value=np.zeros(6)), \
                 patch("droid_plus.datagen.teleop_runner.so101_to_franka", return_value=np.zeros(7)), \
                 patch("droid_plus.datagen.teleop_runner.enforce_min_z", return_value=(np.zeros(7), False)):
                result = run_teleop_episode(
                    config=config, session=session, droid=droid, teleop=Mock(),
                    gripper_initialized=False, pin_model=None, pin_data=None, ee_frame="",
                    recorder=recorder, should_stop=lambda: False, get_phase=phases.get,
                )
            finalize_teleop_episode_recording(result=result, config=config, session=session)
            self.assertEqual(recorder.phase_labels, [0, 1])
            meta = json.loads((Path(recorder.episode_dir) / "meta.json").read_text())
            self.assertTrue(meta["phase_annotation"]["needs_review"])

    def test_ui_phase_endpoint(self):
        import scripts.run_teleop_ui as ui
        from fastapi import HTTPException

        state = SimpleNamespace(status="running", lock=threading.Lock(), phases=PhaseSelector())
        with patch.object(ui, "_state", state):
            self.assertEqual(ui.episode_phase({"phase_index": 2})["phase_index"], 2)
            with self.assertRaises(HTTPException) as error:
                ui.episode_phase({"phase_index": "3"})
            self.assertEqual(error.exception.status_code, 400)
            state.status = "idle"
            with self.assertRaises(HTTPException) as error:
                ui.episode_phase({"phase_index": 3})
            self.assertEqual(error.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
