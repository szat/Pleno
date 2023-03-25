import torch
import unittest
from trilinear_interpolation_torch import trilinear_interpolation


# https://en.wikipedia.org/wiki/Trilinear_interpolation
class TestTrilinearInterpolation(unittest.TestCase):

    def setUp(self):
        # Create a 3D data array to interpolate
        self.c = torch.Tensor([[[0, 0], [0, 0]], [[2, 2], [2, 2]]])
        self.c2 = torch.Tensor([[[[0, 1], [0, 1]], [[0, 1], [0, 1]]], [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]])

    def test_interpolation_inside_box(self):
        # Test interpolation within bounds of the data array
        xyz = torch.Tensor([1.0, 1.0, 1.0])
        expected_value = torch.Tensor([1.0]).reshape([1, 1])
        computed_value = trilinear_interpolation(xyz, self.c, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_inside_box2(self):
        # Test interpolation within bounds of the data array
        xyz = torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Shape n x 1
        expected_value = torch.Tensor([1.0, 1.0]).reshape([2, 1])
        computed_value = trilinear_interpolation(xyz, self.c, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_inside_box_many_colors(self):
        # Test interpolation within bounds of the data array
        xyz = torch.Tensor([1.0, 1.0, 1.0])
        # Shape 1 x c
        expected_value = torch.Tensor([1.0, 2.0]).reshape([1, 2])
        computed_value = trilinear_interpolation(xyz, self.c2, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_inside_box2_many_colors(self):
        # Test interpolation within bounds of the data array
        xyz = torch.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        # Shape n x c
        expected_value = torch.Tensor([[1.0, 2.0], [1.0, 2.0]]).reshape([2, 2])
        computed_value = trilinear_interpolation(xyz, self.c2, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_on_face_nochange(self):
        # Test interpolation on the bounds of the data array
        xyz = torch.Tensor([0.0, 1.0, 1.0])
        expected_value = torch.Tensor([0.0]).reshape([1, 1])
        computed_value = trilinear_interpolation(xyz, self.c, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_on_face_change(self):
        # Test interpolation on the bounds of the data array
        xyz = torch.Tensor([1.0, 1.0, 0.0])
        expected_value = torch.Tensor([1.0]).reshape([1,1])
        computed_value = trilinear_interpolation(xyz, self.c, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_on_edge_nochange(self):
        # Test interpolation on the bounds of the data array
        xyz = torch.Tensor([0.0, 0.0, 0.5])
        expected_value = torch.Tensor([0.0]).reshape([1,1])
        computed_value = trilinear_interpolation(xyz, self.c, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_on_edge_change(self):
        # Test interpolation on the bounds of the data array
        xyz = torch.Tensor([1.0, 0.0, 0.0])
        expected_value = torch.Tensor([1.0]).reshape([1,1])
        computed_value = trilinear_interpolation(xyz, self.c, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)

    def test_interpolation_at_corners(self):
        # Test interpolation at the corners of the data array
        xyz = torch.Tensor([[0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 2.0, 2.0],
                        [0.0, 0.0, 2.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 2.0, 0.0],
                        [2.0, 2.0, 2.0],
                        [2.0, 0.0, 2.0]])
        expected_value = torch.Tensor([0,0,0,0,2,2,2,2]).reshape([8,1])
        computed_value = trilinear_interpolation(xyz, self.c, torch.zeros(3), 2.0, 2.0, 2.0)
        torch.testing.assert_close(expected_value, computed_value)


if __name__=='__main__':
    unittest.main()
