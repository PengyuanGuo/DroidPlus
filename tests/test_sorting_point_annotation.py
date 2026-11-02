"""Real-recording point annotation tests; no model download or GPU required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from scripts.annotate_sorting_points import (
    ground_episode,
    ground_segment,
    read_episode,
    read_rgb,
    sample_indices,
    write_episode,
)


class SortingPointAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.episode = Path(self.temporary.name)
        (self.episode / "left").mkdir()
        self.path = self.episode / "steps.jsonl"
        rows = []
        for obj in ("banana", "apple", "salt bottle"):
            for phase in (0, 1):
                bin_color = "white" if obj == "salt bottle" else "blue"
                prompt = f"grasp {obj}" if phase == 0 else f"place {obj} to the {bin_color} bin"
                for _ in range(32):
                    seq = len(rows)
                    relative = f"left/{seq:09d}.jpg"
                    cv2.imwrite(str(self.episode / relative), np.zeros((60, 80, 3), np.uint8))
                    rows.append({"seq": seq, "phase_index": phase, "phase_prompt": prompt, "images": {"left": relative}})
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.args = SimpleNamespace(
            samples=5, skip_first_frames=2, min_valid_samples=3, roi=[10, 10, 40],
            checkpoint="fake", dtype="bfloat16", load_in_4bit=False, max_image_size=448,
        )

    def test_distinct_segment_points_and_source_preservation(self):
        original = self.path.read_bytes()
        rows, segments = read_episode(self.path)
        self.assertEqual(len(segments), 6)
        model = Mock()

        def point(image, prompt):
            self.assertEqual(image.shape, (40, 40, 3))
            # Different objects get different points despite sharing phase label 0.
            x = 8 if "banana" in prompt else 24 if "apple" in prompt else 16
            return np.array([x, 20], dtype=np.float32)

        model._point.side_effect = point
        results = [ground_segment(model, self.episode, rows, segment, self.args) for segment in segments]
        write_episode(self.path, rows, results, self.args)
        annotated = [json.loads(line) for line in (self.episode / "steps_molmo.jsonl").read_text().splitlines()]
        np.testing.assert_allclose(annotated[0]["point_pixel"], [0.2, 0.5])
        np.testing.assert_allclose(annotated[64]["point_pixel"], [0.6, 0.5])
        self.assertEqual(self.path.read_bytes(), original)
        for old, new in zip((json.loads(s) for s in original.splitlines()), annotated):
            new.pop("point_pixel")
            self.assertEqual(new, old)
        report = json.loads((self.episode / "molmo_point_annotations.json").read_text())
        self.assertFalse(report["needs_review"])
        self.assertEqual(report["roi_xy_size"], [10, 10, 40])
        self.assertEqual(len(list((self.episode / "molmo_previews").glob("*.jpg"))), 6)

    def test_fallback_and_failed_segments(self):
        rows, segments = read_episode(self.path)
        model = Mock()

        def point(image, prompt):
            if "on the table" in prompt:
                raise ValueError("No point")
            return [8, 20]

        model._point.side_effect = point
        result = ground_segment(model, self.episode, rows, segments[0], self.args)
        self.assertFalse(result["needs_review"])
        self.assertTrue(all(sample["errors"] for sample in result["samples"]))
        model._point.side_effect = None
        model._point.return_value = [float("nan"), 5]
        failed = ground_segment(model, self.episode, rows, segments[0], self.args)
        self.assertTrue(failed["needs_review"])
        self.assertEqual(failed["point_pixel"], [-1.0, -1.0])
        model._point.side_effect = RuntimeError("CUDA out of memory")
        with self.assertRaises(RuntimeError):
            ground_segment(model, self.episode, rows, segments[0], self.args)

    def test_reject_bad_roi_and_mismatched_pair(self):
        with self.assertRaises(ValueError):
            read_rgb(self.episode / "left/000000000.jpg", [60, 0, 40])
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        for row in rows[32:64]:
            row["phase_prompt"] = "place apple to the blue bin"
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(ValueError):
            read_episode(self.path)

    def test_spaced_sampling_and_short_phases(self):
        self.assertEqual(sample_indices(100, 200, 5, 2), [102, 107, 113, 118, 124])
        self.assertEqual(sample_indices(10, 11, 5, 2), [10])

    def test_four_phase_targets_and_incomplete_cycle(self):
        rows = []
        for obj in ("banana", "salt bottle"):
            for phase, prefix in enumerate(("pre-grasp", "grasp", "pre-place", "place")):
                rows.append({"seq": len(rows), "phase_index": phase,
                             "phase_prompt": f"{prefix} {obj}",
                             "images": {"left": "left/000000000.jpg"}})
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        _, segments = read_episode(self.path)
        self.assertEqual(len(segments), 8)
        self.assertIn("banana on the table", segments[1]["molmo_prompt"])
        for index in (2, 3):
            self.assertIn("inside of the blue bin", segments[index]["molmo_prompt"])
        for index in (6, 7):
            self.assertIn("inside of the white bin", segments[index]["molmo_prompt"])
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
        with self.assertRaisesRegex(ValueError, "complete four-phase cycles"):
            read_episode(self.path)

    def test_drawer_six_phase_targets_and_mislabelled_frame(self):
        names = ("grasp_handle_open", "open_drawer_and_home", "pick_cube_and_lift",
                 "carry_place_and_home", "grasp_handle_close", "close_drawer_and_home")
        rows = [{"seq": phase, "phase_index": phase, "phase_prompt": name,
                 "images": {"left": "left/000000000.jpg"}} for phase, name in enumerate(names)]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        _, segments = read_episode(self.path)
        self.assertEqual([s["phase_index"] for s in segments], list(range(6)))
        for index in (0, 1, 4, 5):
            self.assertIn("black panel on the front of the white drawer", segments[index]["molmo_prompt"])
        self.assertIn("orange cube", segments[2]["molmo_prompt"])
        self.assertIn("in the drawer", segments[3]["molmo_prompt"])
        # A prompt that disagrees with its phase index is a collection mis-press.
        rows[4]["phase_prompt"] = "close_drawer_and_home"
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "Unsupported phase/prompt"):
            read_episode(self.path)
        rows[4]["phase_prompt"] = names[4]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows[:5]))
        with self.assertRaisesRegex(ValueError, "six drawer phases"):
            read_episode(self.path)

    def test_drawer_handle_window_and_copied_phases(self):
        names = ("grasp_handle_open", "open_drawer_and_home", "pick_cube_and_lift",
                 "carry_place_and_home", "grasp_handle_close", "close_drawer_and_home")
        rows = [{"seq": i, "phase_index": i // 40, "phase_prompt": names[i // 40],
                 "images": {"left": f"left/{i % 192:09d}.jpg"}} for i in range(240)]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        rows, segments = read_episode(self.path)
        args = SimpleNamespace(**{**vars(self.args), "roi": None})
        placing = []

        def point(image, prompt):  # 80x60 frames; clamp at (0.325, 0.30), closed handle at (0.40, 0.30)
            if "placed" in prompt:  # the carried cube early on, the drawer opening once the arm has moved
                placing.append(prompt)
                return np.array([40, 40] if len(placing) <= 6 else [34, 16], dtype=np.float32)
            if "cube" in prompt:
                return np.array([40, 40], dtype=np.float32)
            if "panel" in prompt or "clamp" in prompt or "open drawer" in prompt:
                return np.array([26, 18], dtype=np.float32)  # the primary prompt mistakes the clamp
            # Fallback finds the closed handle in phase 0 and the open handle after placing (phase 4).
            return np.array([36, 24] if placing else [32, 18], dtype=np.float32)

        model = Mock()
        model._point.side_effect = point
        results = ground_episode(model, self.episode, rows, segments, args)
        self.assertFalse(any(r["needs_review"] for r in results))
        np.testing.assert_allclose(results[0]["point_pixel"], [0.4, 0.3])
        self.assertEqual(results[1]["point_pixel"], results[0]["point_pixel"])
        np.testing.assert_allclose(results[3]["point_pixel"], [0.425, 16 / 60], rtol=1e-6)
        self.assertGreaterEqual(results[3]["preview_frame"], max(results[3]["sample_indices"][:6]))
        self.assertGreater(max(results[3]["sample_indices"]), results[3]["end_frame"])
        np.testing.assert_allclose(results[4]["point_pixel"], [0.45, 0.4])
        np.testing.assert_allclose(results[0]["clamp_point"], [0.325, 0.3])
        for phase in (0, 4):
            self.assertTrue(all("black handle" in s["prompt_used"] for s in results[phase]["samples"]))
        self.assertEqual(results[5]["point_pixel"], results[4]["point_pixel"])
        self.assertEqual(results[5]["copied_from_phase"], 4)


if __name__ == "__main__":
    unittest.main()
