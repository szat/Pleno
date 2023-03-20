import numpy as np
from render import *
import unittest


class TestRender(unittest.TestCase):
    def setUp(self):
        self.cam = Camera()

    def assert_Rnpdot_same_o3dRotate(self, vec_from, vec_to, origin=np.zeros(3)):
        return 0

    def test_assert_rotation(self):
        for i in range(10):
            vec_from = np.random.rand(3)
            vec_to = np.random.rand(3)
            origin = np.random.rand(3)

        return True

    def test_rotation_align_random_normalized(self):
        for i in range(10):
            from_vec = np.random.rand(3)
            to_vec = np.random.rand(3)
            from_vec = from_vec / np.linalg.norm(from_vec)
            to_vec = to_vec / np.linalg.norm(to_vec)
            R = rotation_align(from_vec, to_vec)
            np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)
        return True

    def test_rotation_align_opposite_normalized(self):
        from_vec = np.array([1, 0, 0])
        to_vec = np.array([-1, 0, 0])
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)

        from_vec = np.array([0, 1, 0])
        to_vec = np.array([0, -1, 0])
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)

        from_vec = np.array([0, 0, 1])
        to_vec = np.array([0, 0, -1])
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)

        from_vec = np.array([0, 1, 1])
        to_vec = np.array([0, -1, -1])
        from_vec = from_vec / np.linalg.norm(from_vec)
        to_vec = to_vec / np.linalg.norm(to_vec)
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)

        from_vec = np.array([1, 1, 1])
        to_vec = np.array([-1, -1, -1])
        from_vec = from_vec / np.linalg.norm(from_vec)
        to_vec = to_vec / np.linalg.norm(to_vec)
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)
        return True

    def test_rotation_align_identity(self):
        from_vec = np.array([1, 0, 0])
        to_vec = np.array([1, 0, 0])
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)
        from_vec = np.array([1, 0, 0])
        to_vec = np.array([0, 1, 0])
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)
        return True

    def test_rotation_align_perp(self):
        from_vec = np.array([1, 0, 0])
        to_vec = np.array([0, 1, 0])
        R = rotation_align(from_vec, to_vec)
        np.testing.assert_allclose(np.dot(np.dot(R, from_vec), to_vec), 1.0)
        return True

    def test_rotation_align_not_normalized(self):
        return True

    def test_rotation_align_many_on_first_vec(self):
        return True

    def test_rotation_align_many_on_second_vec(self):
        return True

    def test_rotation_align_many_on_both_vec(self):
        return True

    def test_viz_one_camera(self):
        return True

    def test_viz_rays(self):
        return True


if __name__ == '__main__':
    unittest.main()