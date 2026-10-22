import unittest
from unittest.mock import patch

import numpy as np

from ros2_calib.lidar_cleaner import FisheyeLiDARCleaner


class CenterProjectionModel:
    def project_points(self, points):
        return np.full((len(points), 2), 50.0)


class TestFisheyeLiDARCleaner(unittest.TestCase):
    def test_uses_perspective_face_for_every_fisheye_direction(self):
        points = np.array(
            [
                [1, -1, 0, 0, 0, 0],
                [0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 1, -1],
            ],
            dtype=np.float32,
        )
        extrinsics = np.column_stack((np.eye(3), np.zeros(3)))

        with patch(
            "ros2_calib.lidar_cleaner.LiDARCleaner.run",
            return_value=np.ones(points.shape[1], dtype=bool),
        ) as run:
            mask = FisheyeLiDARCleaner(
                CenterProjectionModel(),
                extrinsics,
                points,
                height=100,
                width=100,
                fov_limit_deg=180,
            ).run()

        np.testing.assert_array_equal(mask, np.ones(points.shape[1], dtype=bool))
        self.assertEqual(run.call_count, 6)

    def test_applies_fisheye_field_of_view_before_cleaning(self):
        points = np.array([[0, 0], [0, 0], [1, -1]], dtype=np.float32)
        extrinsics = np.column_stack((np.eye(3), np.zeros(3)))

        with patch(
            "ros2_calib.lidar_cleaner.LiDARCleaner.run",
            return_value=np.ones(points.shape[1], dtype=bool),
        ):
            mask = FisheyeLiDARCleaner(
                CenterProjectionModel(),
                extrinsics,
                points,
                height=100,
                width=100,
                fov_limit_deg=105,
            ).run()

        np.testing.assert_array_equal(mask, [True, False])


if __name__ == "__main__":
    unittest.main()
