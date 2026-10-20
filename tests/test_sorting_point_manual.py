"""Manual clicks share the correction format and change only the chosen segment."""
from __future__ import annotations

import io
import json
import unittest
from unittest.mock import Mock

from scripts.annotate_sorting_points_correction import prepare_correction
from scripts.annotate_sorting_points_manual import clicked_prediction, make_handler, save_clicked_point
from tests import test_sorting_point_correction as correction_tests


class ManualPointCorrectionTests(unittest.TestCase):
    setUp = correction_tests.SortingPointCorrectionTests.setUp

    def test_normalization_and_invalid_clicks(self):
        self.assertEqual(clicked_prediction([10, 15], 20, 20)["point_pixel"], [0.5, 0.75])
        self.assertEqual(clicked_prediction([0, 0], 20, 20)["point_pixel"], [0, 0])
        self.assertEqual(clicked_prediction([19, 19], 20, 20)["point_pixel"], [0.95, 0.95])
        for click in (None, [], [1], [-1, 5], [20, 5], [5, 20], [True, 5], [float("nan"), 5], [5, float("inf")]):
            with self.subTest(click=click), self.assertRaises(ValueError):
                clicked_prediction(click, 20, 20)

    def test_manual_click_crop_coordinates_and_preserved_segments(self):
        # Another segment still needs review; correcting segment 04 must not clear it.
        report_path = self.episode / "molmo_point_annotations.json"
        report = json.loads(report_path.read_text())
        report["segments"][0]["needs_review"] = True
        report["needs_review"] = True
        report_path.write_text(json.dumps(report))
        context = prepare_correction(self.episode / "molmo_previews/segment_04.jpg", 4)
        source = self.source.read_bytes()
        annotated_path = self.episode / "steps_molmo.jsonl"
        original_lines = annotated_path.read_bytes().splitlines(keepends=True)
        prediction, backup = save_clicked_point(context, [10, 15])
        self.assertEqual(prediction["point_pixel"], [0.5, 0.75])
        self.assertTrue(backup.exists())
        self.assertEqual(self.source.read_bytes(), source)
        report = json.loads(report_path.read_text())
        self.assertTrue(report["needs_review"])
        segment = report["segments"][4]
        self.assertEqual(segment["method"], "human_click_correction")
        self.assertTrue(segment["human_verified"])
        self.assertFalse(segment["needs_review"])
        self.assertEqual(segment["point_pixel"], [0.5, 0.75])
        self.assertEqual(segment["samples"][0]["prompt_used"], "human_click")
        self.assertEqual(report["corrections"][-1]["method"], "human_click_correction")
        for index, (old, new) in enumerate(zip(original_lines, annotated_path.read_bytes().splitlines(keepends=True))):
            if 16 <= index < 20:
                self.assertEqual(json.loads(new)["point_pixel"], [0.5, 0.75])
            else:
                self.assertEqual(old, new)
        for index in (0, 1, 2, 3, 5):
            self.assertEqual(report["segments"][index], context["report"]["segments"][index])

    def handler(self, context, state, path, payload, token="test-token"):
        cls = make_handler(context, state, "test-token", b"page", b"image")
        handler = cls.__new__(cls)
        data = json.dumps(payload).encode()
        handler.path = path
        handler.headers = {"Content-Length": str(len(data)), "X-Annotation-Token": token}
        handler.rfile = io.BytesIO(data)
        handler.reply = Mock()
        return handler

    def test_cancel_and_invalid_session_do_not_write(self):
        context = prepare_correction(self.image)
        original = {p: p.read_bytes() for p in self.episode.rglob("*") if p.is_file()}
        state = {"done": False, "result": None}
        request = self.handler(context, state, "/save", {"pixel_xy": [10, 15]}, token="wrong")
        request.do_POST()
        self.assertEqual(request.reply.call_args.args[0], 403)
        self.assertFalse(state["done"])
        request = self.handler(context, state, "/save", {"pixel_xy": [999, 0]})
        request.do_POST()
        self.assertEqual(request.reply.call_args.args[0], 400)
        self.assertFalse(state["done"])
        request = self.handler(context, state, "/cancel", {})
        request.do_POST()
        self.assertEqual(request.reply.call_args.args[0], 200)
        self.assertTrue(state["done"])
        self.assertTrue(state["result"]["cancelled"])
        self.assertEqual(original, {p: p.read_bytes() for p in self.episode.rglob("*") if p.is_file()})

    def test_save_from_browser_route(self):
        context = prepare_correction(self.image, 3)
        state = {"done": False, "result": None}
        request = self.handler(context, state, "/save", {"pixel_xy": [10, 15]})
        request.do_POST()
        self.assertEqual(request.reply.call_args.args[0], 200)
        self.assertEqual(json.loads(request.reply.call_args.args[1])["point_pixel"], [0.5, 0.75])
        self.assertTrue(state["done"])
        request.do_POST()
        self.assertEqual(request.reply.call_args.args[0], 409)


if __name__ == "__main__":
    unittest.main()
