import numpy as np
import unittest
from trilinear_interpolation import *


# https://en.wikipedia.org/wiki/Trilinear_interpolation
class TestTrilinearInterpolation(unittest.TestCase):

    def setUp(self):
        # Create a 3D data array to interpolate
        self.c = np.array([[[0, 0], [0, 0]], [[2, 2], [2, 2]]])
        self.c2 = np.array([[[[0, 1], [0, 1]], [[0, 1], [0, 1]]], [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]])

    def test_interpolation_inside_box(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([1.0, 1.0, 1.0])
        expected_value = np.array([1.0]).reshape([1, 1])
        computed_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        self.assertAlmostEqual(expected_value, computed_value, places=10)

    def test_interpolation_inside_box2(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Shape n x 1
        expected_value = np.array([1.0, 1.0]).reshape([2, 1])
        computed_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_inside_box_many_colors(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([1.0, 1.0, 1.0])
        # Shape 1 x c
        expected_value = np.array([1.0, 2.0]).reshape([1, 2])
        computed_value = trilinear_interpolation(xyz, self.c2, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_inside_box2_many_colors(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Shape n x c
        expected_value = np.array([[1.0, 2.0], [1.0, 2.0]]).reshape([2, 2])
        computed_value = trilinear_interpolation(xyz, self.c2, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_on_face_nochange(self):
        # Test interpolation on the bounds of the data array
        xyz = np.array([0.0, 1.0, 1.0])
        expected_value = np.array([0.0]).reshape([1, 1])
        computed_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_on_face_change(self):
        # Test interpolation on the bounds of the data array
        xyz = np.array([1.0, 1.0, 0.0])
        expected_value = np.array([1.0]).reshape([1,1])
        computed_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_on_edge_nochange(self):
        # Test interpolation on the bounds of the data array
        xyz = np.array([0.0, 0.0, 0.5])
        expected_value = np.array([0.0]).reshape([1,1])
        computed_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_on_edge_change(self):
        # Test interpolation on the bounds of the data array
        xyz = np.array([1.0, 0.0, 0.0])
        expected_value = np.array([1.0]).reshape([1,1])
        computed_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_at_corners(self):
        # Test interpolation at the corners of the data array
        xyz = np.array([[0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 2.0, 2.0],
                        [0.0, 0.0, 2.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [2.0, 2.0, 2.0],
                        [2.0, 0.0, 2.0]])
        expected_value = np.array([0,0,0,0,2,2,2,2]).reshape([8,1])
        computed_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_dot_at_corners(self):
        # Test interpolation at the corners of the data array
        xyz = np.array([[0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 2.0, 2.0],
                        [0.0, 0.0, 2.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [2.0, 2.0, 2.0],
                        [2.0, 0.0, 2.0]])
        expected_value = np.array([0,0,0,0,2,2,2,2]).reshape([8,1])
        computed_value = trilinear_interpolation_dot(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation_dot_random(self):
        # Test interpolation at the corners of the data array
        xyz = np.random.rand(10, 3)
        expected_value = trilinear_interpolation(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        computed_value = trilinear_interpolation_dot(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation2_inside_box(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([1.0, 1.0, 1.0])
        expected_value = np.array([1.0]).reshape([1, 1])
        computed_value = trilinear_interpolation_dot(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        self.assertAlmostEqual(expected_value, computed_value, places=10)

    def test_interpolation2_inside_box2(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Shape n x 1
        expected_value = np.array([1.0, 1.0]).reshape([2, 1])
        computed_value = trilinear_interpolation_dot(xyz, self.c, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation2_inside_box_many_colors(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([1.0, 1.0, 1.0])
        # Shape 1 x c
        expected_value = np.array([1.0, 2.0]).reshape([1, 2])
        computed_value = trilinear_interpolation_dot(xyz, self.c2, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_interpolation2_inside_box2_many_colors(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Shape n x c
        expected_value = np.array([[1.0, 2.0], [1.0, 2.0]]).reshape([2, 2])
        computed_value = trilinear_interpolation_dot(xyz, self.c2, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_trilinear_coefficients_two_entries(self):
        xyz = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Shape n x c
        expected_value = np.array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                                   [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]])
        computed_value = trilinear_coefficients(xyz, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)

    def test_trilinear_coefficients_one_entry(self):
        # Test interpolation within bounds of the data array
        xyz = np.array([1.0, 1.0, 1.0])
        expected_value = np.array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]])
        computed_value = trilinear_coefficients(xyz, np.zeros(3), 2.0, 2.0, 2.0)
        np.testing.assert_array_almost_equal(expected_value, computed_value)


if __name__=='__main__':
    unittest.main()