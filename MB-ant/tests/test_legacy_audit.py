import unittest

import numpy as np

from audit_dataset import distance_to_polyline


class GeometryTest(unittest.TestCase):
    def test_distance_uses_nearest_segment_not_nearest_vertex(self):
        route = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
        points = np.array([[1.0, 0.0], [1.0, 0.5], [2.5, 1.0]])
        np.testing.assert_allclose(distance_to_polyline(points, route), [0, 0.5, 0.5])

    def test_duplicate_route_points_are_safe(self):
        route = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(
            distance_to_polyline(np.array([[0.5, 1.0]]), route), [1.0]
        )


if __name__ == "__main__":
    unittest.main()
