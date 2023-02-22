import numpy as np
import unittest
from scipy.special import *
# from renderer import *
from spherical_harmonics import *

# https://en.wikipedia.org/wiki/Trilinear_interpolation
class TestRenderer(unittest.TestCase):
    def setUp(self):
        pi = np.pi
        self.angles = np.array([[pi/4, pi/4],
                           [pi/2, pi/2],
                           [3*pi/4, 3*pi/4],
                           [5*pi/6, pi],
                            [pi / 6, pi],
                            [pi/4, 7*pi/5],
                            [pi, pi/4], [pi, 5*pi/4], [pi, 2*pi]])

    def spherical_to_cartesian(self, theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z])

    def test_sh_spherical_shape_one_entry(self):
        theta, phi = self.angles[0,0], self.angles[0,1]
        res = sh_spherical(theta, phi)
        np.testing.assert_array_almost_equal(res.shape, np.array([1,9]))

    def test_sh_cartesian_shape_one_entry(self):
        xyz = np.random.rand(3)
        res = sh_cartesian(xyz)
        np.testing.assert_array_almost_equal(res.shape, np.array(9))

    def test_sh_cartesian_shape_one_entry_vec(self):
        xyz = np.random.rand(1, 3)
        res = sh_cartesian(xyz)
        np.testing.assert_array_almost_equal(res.shape, np.array([1, 9]))

    def test_sh_cartesian_shape_many_entry(self):
        xyz = np.random.rand(4, 3)
        res = sh_cartesian(xyz)
        np.testing.assert_array_almost_equal(res.shape, np.array([4,9]))

    def test_sh_cartesian_shape_many_entry(self):
        xyz = np.random.rand(4, 3)
        res = sh_cartesian(xyz)
        for i in range(4):
            res2 = sh_cartesian(xyz[i, :])
            np.testing.assert_array_almost_equal(res[i], res2)

    def test_sh_different_formulas(self):
        for i in range(len(self.angles)):
            theta, phi = self.angles[i, 0], self.angles[i, 1]
            res1 = sh_spherical(theta, phi)
            res1 = res1[0]
            res1[np.abs(res1) < 1e-08] = 0
            # Note we reverse phi and theta for scipy (idiots)
            res2 = real_sph_harm_vec_scipy(phi, theta)
            res2[np.abs(res2) < 1e-08] = 0
            res3 = sh_cartesian(self.spherical_to_cartesian(theta, phi))
            res3[np.abs(res3) < 1e-08] = 0
            res4 = sh_gritty_book(theta, phi)
            res4[np.abs(res4) < 1e-08] = 0
            np.testing.assert_array_almost_equal(res1, res2)
            np.testing.assert_array_almost_equal(res2, res3)
            np.testing.assert_array_almost_equal(res3, res4)


if __name__=='__main__':
    unittest.main()