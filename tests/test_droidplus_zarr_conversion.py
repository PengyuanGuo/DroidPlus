"""End-to-end native-recording conversion and training-cache compatibility."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import zarr

from scripts.convert_droidplus_to_zarr import convert, encode_text, preflight, transform_points


class DroidPlusZarrConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "recordings"
        self.first = self.make_episode(0)
        self.second = self.make_episode(1)
        self.output = self.root / "converted.zarr"

    def make_episode(self, index):
        episode = self.dataset / "teleop_test" / f"episode_{index:03d}"
        episode.mkdir(parents=True)
        for camera in ("left", "wrist"):
            (episode / camera).mkdir()
        rows, segments = [], []
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.arange(12) * 10
        rgb[:, :, 1] = np.arange(8)[:, None] * 20
        for obj in ("banana", "apple", "salt bottle"):
            for phase in (0, 1):
                prompt = f"grasp {obj}" if phase == 0 else f"place {obj} to the {'white' if obj == 'salt bottle' else 'blue'} bin"
                start = len(rows)
                for _ in range(3):
                    seq = len(rows)
                    images = {}
                    for camera, blue in (("left", 50), ("wrist", 100)):
                        image = rgb.copy()
                        image[:, :, 2] = blue
                        relative = f"{camera}/{seq:09d}.png"
                        cv2.imwrite(str(episode / relative), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        images[camera] = relative
                    rows.append({
                        "seq": seq, "t_wall_s": 1000 + index * 100 + seq * 0.2,
                        "state": {"positions": [0, 1, 2, 3, 4, 5, 6, 0.25]},
                        "action": {"positions": [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, float(phase == 0)]},
                        "images": images, "phase_index": phase, "phase_prompt": prompt,
                        "point_pixel": [0.25, 0.25],
                    })
                segments.append({"phase_index": phase, "phase_prompt": prompt, "start_frame": start, "end_frame": len(rows), "point_pixel": [0.25, 0.25], "needs_review": False})
        raw = [{k: v for k, v in row.items() if k != "point_pixel"} for row in rows]
        raw_text = "".join(json.dumps(row) + "\n" for row in raw)
        (episode / "steps.jsonl").write_text(raw_text)
        (episode / "steps_molmo.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (episode / "molmo_point_annotations.json").write_text(json.dumps({
            "roi_xy_size": None, "source_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
            "needs_review": False, "segments": segments,
        }))
        (episode / "meta.json").write_text(json.dumps({
            "instruction": "sort fruit into blue bin", "step_count": 999,
            "control": {"record_rate_hz": 15},
        }))
        return episode

    def test_complete_schema_caches_points_actions_and_boundaries(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.dataset.rglob("*") if p.is_file()}
        episodes = preflight(self.dataset)
        manifest = convert(episodes, self.output, chunk_frames=4)
        root = zarr.open_group(str(self.output), mode="r")
        self.assertEqual(json.loads((self.output / ".zgroup").read_text())["zarr_format"], 2)
        np.testing.assert_array_equal(root["meta/episode_ends"][:], [18, 36])
        for key, shape in (("state", (36, 8)), ("action", (36, 8)), ("point_pixel", (36, 2)),
                           ("prompt", (36, 128)), ("phase_prompt", (36, 128)), ("phase_label", (36,))):
            self.assertEqual(root[f"data/{key}"].shape, shape)
        self.assertEqual(root["data/phase_label"].dtype, np.dtype("int8"))
        self.assertEqual(root["data/state"].dtype, np.dtype("float32"))
        self.assertEqual(root["data/agent_view"].chunks[0], 4)
        self.assertEqual(root["data/state"].compressor.cname, "zstd")
        self.assertEqual(root["data/state"].compressor.clevel, 5)
        for camera, blue in (("agent_view", 50), ("wrist_cam", 100)):
            hwc = root[f"data/{camera}"][:]
            chw = root[f"_cache/224x224/{camera}.zarr/images"][:]
            self.assertEqual(hwc.shape, (36, 224, 224, 3))
            self.assertEqual(chw.shape, (36, 3, 224, 224))
            np.testing.assert_array_equal(chw, hwc.transpose(0, 3, 1, 2))
            np.testing.assert_array_equal(hwc[0, 0, 0], [20, 0, blue])
        # Raw point (.25*12,.25*8)=(3,2); centered crop is x=2,y=0,size=8.
        np.testing.assert_allclose(root["data/point_pixel"][:], np.tile([0.125, 0.25], (36, 1)))
        np.testing.assert_allclose(root["data/state"][:, 7], 0.75)
        np.testing.assert_allclose(root["data/action"][:3, 7], 0)
        np.testing.assert_allclose(root["data/action"][3:6, 7], 1)
        np.testing.assert_allclose(root["data/action"][0, :7], np.arange(7) + 0.1)
        self.assertEqual(bytes(root["data/prompt"][0]).rstrip(b"\0").decode(), "sort fruit into blue bin")
        np.testing.assert_allclose(root["data/timestamp"][:3], [0, 0.2, 0.4])
        self.assertEqual(manifest["n_episodes"], 2)
        self.assertAlmostEqual(manifest["observed_fps_median"], 5)
        self.assertEqual(root.attrs["agent_roi_xy_size"], [2, 0, 8])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.dataset.rglob("*") if p.is_file()})

    def test_roi_transform_from_annotation_crop_and_invalid_points(self):
        point = transform_points(np.array([[0.5, 0.5]]), [3, 0, 4], [2, 0, 8], 12, 8)
        np.testing.assert_allclose(point, [[0.375, 0.25]])
        with self.assertRaisesRegex(ValueError, "outside output ROI"):
            transform_points(np.array([[0, 0.5]]), None, [2, 0, 8], 12, 8)
        for point in ([[-1, -1]], [[float("nan"), 0.5]], [[2, 0.5]]):
            with self.assertRaises(ValueError):
                transform_points(np.asarray(point), None, [2, 0, 8], 12, 8)

    def test_review_flags_and_stale_raw_metadata(self):
        report_path = self.first / "molmo_point_annotations.json"
        report = json.loads(report_path.read_text())
        report["segments"][0]["needs_review"] = True
        report_path.write_text(json.dumps(report))
        # Raw JSON is not used for data after annotation; keep corrected JSONL authoritative.
        (self.first / "steps.jsonl").write_text("raw file changed after annotation")
        episodes = preflight(self.dataset)
        self.assertEqual(int(episodes[0]["data"]["point_needs_review"].sum()), 3)
        self.assertTrue(any("hash differs" in x for x in episodes[0]["manifest"]["notes"]))
        self.assertTrue(any("step_count is stale" in x for x in episodes[0]["manifest"]["notes"]))
        with self.assertRaisesRegex(ValueError, "still marked needs_review"):
            preflight(self.dataset, require_reviewed=True)

    def test_overwrite_backups_and_bad_image_preserves_existing_store(self):
        convert(preflight(self.dataset, max_episodes=1), self.output, chunk_frames=4)
        original_manifest = (self.output / "conversion_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            convert(preflight(self.dataset), self.output)
        convert(preflight(self.dataset), self.output, chunk_frames=4, overwrite=True)
        backups = list(self.root.glob("converted.zarr.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "conversion_manifest.json").read_bytes(), original_manifest)
        current_manifest = (self.output / "conversion_manifest.json").read_bytes()
        (self.first / "left/000000001.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Cannot decode image"):
            convert(preflight(self.dataset), self.output, chunk_frames=4, overwrite=True)
        self.assertEqual((self.output / "conversion_manifest.json").read_bytes(), current_manifest)

    def test_report_mismatch_and_unannotated_episodes_rejected(self):
        path = self.first / "steps_molmo.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["point_pixel"] = [0.5, 0.5]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "disagree with annotation report"):
            preflight(self.dataset)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing corrected annotations"):
            preflight(self.dataset)

    def test_utf8_truncation(self):
        encoded = encode_text("é" * 100)
        self.assertEqual(encoded.shape, (128,))
        self.assertEqual(bytes(encoded).decode("utf-8"), "é" * 64)


if __name__ == "__main__":
    unittest.main()
