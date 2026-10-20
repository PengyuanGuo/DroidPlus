"""Point corrections must affect one contiguous segment and preserve backups."""
from __future__ import annotations

import copy
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from scripts.annotate_sorting_points import read_episode, write_episode
from scripts.annotate_sorting_points_correction import (
    apply_correction,
    correction_prompts,
    predict_point,
    prepare_correction,
)


class SortingPointCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.episode = Path(self.temporary.name)
        (self.episode / "left").mkdir()
        rows = []
        for obj in ("banana", "orange", "salt bottle"):
            for phase in (0, 1):
                prompt = f"grasp {obj}" if phase == 0 else f"place {obj} to the {'white' if obj == 'salt bottle' else 'blue'} bin"
                for _ in range(4):
                    seq = len(rows)
                    image = f"left/{seq:09d}.jpg"
                    cv2.imwrite(str(self.episode / image), np.zeros((30, 40, 3), np.uint8))
                    rows.append({"seq": seq, "images": {"left": image}, "phase_index": phase, "phase_prompt": prompt})
        self.source = self.episode / "steps.jsonl"
        self.source.write_text("".join(json.dumps(r) + "\n" for r in rows))
        rows, segments = read_episode(self.source)
        results = [{**s, "sample_indices": [s["start_frame"]], "point_pixel": [0.1, 0.2], "needs_review": False} for s in segments]
        self.settings = {"checkpoint": "fake-model", "dtype": "bfloat16", "load_in_4bit": False, "max_image_size": 448, "roi_xy_size": [10, 5, 20]}
        args = SimpleNamespace(**{k: v for k, v in self.settings.items() if k != "roi_xy_size"}, roi=[10, 5, 20])
        write_episode(self.source, rows, results, args)
        self.image = self.episode / "left/000000014.jpg"

    def test_correct_one_segment_preserve_other_rows_and_previews(self):
        context = prepare_correction(self.image, expected_segment=3)
        self.assertEqual(context["segment_index"], 3)
        self.assertEqual(context["rgb"].shape, (20, 20, 3))
        original = {p: p.read_bytes() for p in self.episode.rglob("*") if p.is_file()}
        previous_segments = copy.deepcopy(context["report"]["segments"])
        model = Mock()
        model._point.return_value = [10, 15]
        prediction = predict_point(model, context["rgb"], ["Point to the blue bin"])
        self.assertEqual(prediction["point_pixel"], [0.5, 0.75])
        backup = apply_correction(context, prediction, self.settings)
        self.assertEqual(self.source.read_bytes(), original[self.source])
        annotated = self.episode / "steps_molmo.jsonl"
        before = original[annotated].decode().splitlines(keepends=True)
        after = annotated.read_text().splitlines(keepends=True)
        for i, (old, new) in enumerate(zip(before, after)):
            if 12 <= i < 16:
                row = json.loads(new)
                self.assertEqual(row.pop("point_pixel"), [0.5, 0.75])
                old_row = json.loads(old)
                old_row.pop("point_pixel")
                self.assertEqual(row, old_row)
            else:
                self.assertEqual(old, new)
        report = json.loads((self.episode / "molmo_point_annotations.json").read_text())
        self.assertTrue(report["needs_review"])
        self.assertEqual(report["segments"][3]["sample_indices"], [14])
        self.assertEqual(report["segments"][3]["point_pixel"], [0.5, 0.75])
        self.assertEqual(report["corrections"][0]["previous_segment"], previous_segments[3])
        for i in (0, 1, 2, 4, 5):
            self.assertEqual(report["segments"][i], previous_segments[i])
            preview = self.episode / f"molmo_previews/segment_{i:02d}.jpg"
            self.assertEqual(preview.read_bytes(), original[preview])
        with tarfile.open(backup, "r:gz") as archive:
            for filename in ("steps_molmo.jsonl", "molmo_point_annotations.json", "molmo_previews/segment_03.jpg"):
                self.assertEqual(archive.extractfile(filename).read(), original[self.episode / filename])

    def test_wrong_segment_and_changed_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "not requested segment"):
            prepare_correction(self.image, expected_segment=1)
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed since annotation"):
            prepare_correction(self.image)

    def test_failed_prediction_does_not_change_files(self):
        context = prepare_correction(self.image)
        before = {p: p.read_bytes() for p in self.episode.rglob("*") if p.is_file()}
        model = Mock()
        model._point.side_effect = [[-1, 0], [float("nan"), 5]]
        with self.assertRaisesRegex(ValueError, "All correction prompts failed"):
            predict_point(model, context["rgb"], ["Primary", "Fallback"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.episode.rglob("*") if p.is_file()})

    def test_concurrent_edit_rejected_without_overwriting(self):
        context = prepare_correction(self.image)
        annotated = self.episode / "steps_molmo.jsonl"
        changed = annotated.read_bytes() + b"\n"
        annotated.write_bytes(changed)
        prediction = {"point_pixel": [0.5, 0.75], "pixel_xy": [10, 15], "errors": [], "prompt_used": "Point to the blue bin"}
        with self.assertRaisesRegex(ValueError, "File changed during correction"):
            apply_correction(context, prediction, self.settings)
        self.assertEqual(annotated.read_bytes(), changed)

    def test_fallback_and_cuda_errors(self):
        model = Mock()
        model._point.side_effect = [ValueError("No point"), [10, 15]]
        result = predict_point(model, np.zeros((20, 20, 3), np.uint8), ["Primary", "Fallback"])
        self.assertEqual(result["prompt_used"], "Fallback")
        self.assertEqual(len(result["errors"]), 1)
        model._point.side_effect = RuntimeError("CUDA out of memory")
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            predict_point(model, np.zeros((20, 20, 3), np.uint8), ["Primary", "Fallback"])

    def test_preview_resolves_to_original_without_annotations(self):
        preview = self.episode / "molmo_previews/segment_04.jpg"
        context = prepare_correction(preview, expected_segment=4)
        self.assertEqual(context["image"], self.episode / "left/000000016.jpg")
        self.assertEqual(context["requested_image"], preview)
        self.assertEqual(context["segment_index"], 4)
        # Raw fixture is black; the preview contains colored cross/text.
        self.assertTrue(np.all(context["rgb"] == 0))
        with self.assertRaisesRegex(ValueError, "not requested segment"):
            prepare_correction(preview, expected_segment=3)

    def test_custom_target_does_not_fall_back_to_old_object(self):
        context = prepare_correction(self.episode / "molmo_previews/segment_04.jpg")
        prompts = correction_prompts(context["segment"], "Point to the blue bottle", None)
        self.assertEqual(prompts, ["Point to the blue bottle"])
        self.assertEqual(correction_prompts(context["segment"], None, None),
                         ["Point to the salt bottle on the table", "Point to the salt bottle"])
        self.assertEqual(correction_prompts(context["segment"], "Point to the blue bottle", "Point to the bottle"),
                         ["Point to the blue bottle", "Point to the bottle"])
        model = Mock()
        model._point.return_value = [10, 15]
        prediction = predict_point(model, context["rgb"], prompts)
        apply_correction(context, prediction, self.settings)
        report = json.loads((self.episode / "molmo_point_annotations.json").read_text())
        segment = report["segments"][4]
        self.assertEqual(segment["molmo_prompt"], "Point to the blue bottle")
        self.assertEqual(segment["fallback_prompt"], "")
        self.assertEqual(segment["samples"][0]["prompt_used"], "Point to the blue bottle")
        self.assertEqual(segment["phase_prompt"], "grasp salt bottle")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            correction_prompts(context["segment"], " ", None)


if __name__ == "__main__":
    unittest.main()
