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

    def test_rotation_align_random_normalized_parallel(self):
        nb = 10
        from_vec = np.random.rand(nb, 3)
        to_vec = np.random.rand(nb, 3)
        from_vec = from_vec / np.expand_dims(np.linalg.norm(from_vec, axis=1), 1)
        to_vec = to_vec / np.expand_dims(np.linalg.norm(to_vec, axis=1), 1)
        Rs = rotation_align(from_vec, to_vec)
        assert Rs.shape == (nb, 3, 3)
        for i in range(nb):
            np.testing.assert_allclose(np.dot(np.dot(Rs[i], from_vec[i]), to_vec[i]), 1.0)
        return True

    def test_rotation_align_opposite_normalized_parallel(self):
        nb = 6
        from_vec = np.empty([nb, 3])
        to_vec = np.empty([nb, 3])

        from_vec[0] = np.array([1, 0, 0])
        to_vec[0] = np.array([-1, 0, 0])

        from_vec[1] = np.array([0, 1, 0])
        to_vec[1] = np.array([0, -1, 0])

        from_vec[2] = np.array([0, 0, 1])
        to_vec[2] = np.array([0, 0, -1])

        from_vec[3] = np.array([0, 1, 1])
        to_vec[3] = np.array([0, -1, -1])

        from_vec[4] = np.array([1, 1, 1])
        to_vec[4] = np.array([-1, -1, -1])

        from_vec[5] = np.array([1, 1, 1])
        to_vec[5] = np.array([1, 1, 1])

        from_vec = from_vec / np.expand_dims(np.linalg.norm(from_vec, axis=1), 1)
        to_vec = to_vec / np.expand_dims(np.linalg.norm(to_vec, axis=1), 1)

        Rs = rotation_align(from_vec, to_vec)

        assert Rs.shape == (nb, 3, 3)
        for i in range(nb):
            np.testing.assert_allclose(np.dot(np.dot(Rs[i], from_vec[i]), to_vec[i]), 1.0)
        return True

    def test_rotation_align_perp_parallel(self):
        nb = 2
        from_vec = np.empty([nb, 3])
        to_vec = np.empty([nb, 3])

        from_vec[0] = np.array([1, 0, 0])
        to_vec[0] = np.array([0, 1, 0])

        from_vec[1] = np.array([0, 1, 0])
        to_vec[1] = np.array([0, 0, -1])
        Rs = rotation_align(from_vec, to_vec)

        assert Rs.shape == (nb, 3, 3)
        for i in range(nb):
            np.testing.assert_allclose(np.dot(np.dot(Rs[i], from_vec[i]), to_vec[i]), 1.0)
        return True

    def test_viz_one_camera(self):
        return True

    def test_viz_rays(self):
        return True


if __name__ == '__main__':
    unittest.main()