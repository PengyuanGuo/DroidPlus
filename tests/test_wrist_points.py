from __future__ import annotations

import unittest

import numpy as np

from scripts.convert_droidplus_to_zarr import map_wrist_points


class WristPointMappingTest(unittest.TestCase):
    """Wrist points go from the raw 672x376 wrist image into the converter's wrist_cam crop."""

    def test_maps_into_centred_crop_and_hides_points_outside(self):
        roi = [148, 0, 376]  # centred square
        rows = [
            {"wrist_point_pixel": [0.5, 0.5], "wrist_point_visible": True},  # image centre -> crop centre
            {"wrist_point_pixel": [148 / 672, 0.0], "wrist_point_visible": True},  # crop top-left corner
            {"wrist_point_pixel": [0.05, 0.5], "wrist_point_visible": True},  # left of the crop
            {"wrist_point_pixel": [-1.0, -1.0], "wrist_point_visible": False},  # not visible
        ]
        xy, vis = map_wrist_points(rows, roi, (376, 672, 3))
        np.testing.assert_allclose(xy[0], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(xy[1], [0.0, 0.0], atol=1e-6)
        self.assertEqual(vis.tolist(), [True, True, False, False])
        np.testing.assert_array_equal(xy[2:], -1.0)

    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            map_wrist_points([{"wrist_point_pixel": [float("nan"), 0.1], "wrist_point_visible": True}], [148, 0, 376], (376, 672, 3))


if __name__ == "__main__":
    unittest.main()
