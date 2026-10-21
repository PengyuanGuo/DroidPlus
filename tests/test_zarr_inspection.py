"""Validate readable training data and detect stale/corrupt converted arrays."""
from __future__ import annotations

import unittest

import zarr

from scripts.convert_droidplus_to_zarr import convert, preflight
from scripts.verify_droidplus_zarr import verify
from tests import test_droidplus_zarr_conversion as conversion_tests


class ZarrInspectionTests(unittest.TestCase):
    setUp = conversion_tests.DroidPlusZarrConversionTests.setUp
    make_episode = conversion_tests.DroidPlusZarrConversionTests.make_episode

    def test_full_validation_with_sources(self):
        convert(preflight(self.dataset), self.output, chunk_frames=4)
        result = verify(self.output, check_sources=True, full_images=True)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["frames"], 36)

    def test_corrupted_point_and_cache_are_detected(self):
        convert(preflight(self.dataset), self.output, chunk_frames=4)
        root = zarr.open_group(str(self.output), mode="a")
        root["data/point_pixel"][0] = [-1, -1]
        root["_cache/224x224/agent_view.zarr/images"][0, 0, 0, 0] = 255
        result = verify(self.output, full_images=True)
        self.assertFalse(result["passed"])
        self.assertTrue(any("sentinel" in message for message in result["errors"]))
        self.assertTrue(any("Image/cache mismatch" in message for message in result["errors"]))

    def test_annotations_edited_after_conversion_are_detected(self):
        convert(preflight(self.dataset), self.output, chunk_frames=4)
        corrected = self.first / "steps_molmo.jsonl"
        corrected.write_bytes(corrected.read_bytes() + b"\n")
        result = verify(self.output, check_sources=True)
        self.assertFalse(result["passed"])
        self.assertTrue(any("corrected source changed" in message for message in result["errors"]))


if __name__ == "__main__":
    unittest.main()
