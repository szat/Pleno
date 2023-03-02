import numpy as np
import unittest
from scipy.special import *
# from renderer import *
from frustum_branch import *


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 5
        self.diff = 1.0
        # So the grid goes like [0.0, 1.0, ... , 4.0]
        self.nb_rays = 7
        self.nb_samples = 11

    def test_rays_to_frustrum(self):
        R = np.random.rand(self.nb_rays, 6)
        grid = np.random.rand([self.idim, self.idim, self.idim, 10]) # first one for occlusion, between [0,1] the rest for harmonics
        F = rays_to_frustrum(R, grid)
        np.testing.assert_array_almost_equal(F.shape, np.array([self.nb_rays, self.idim, self.idim, self.idim, 3]))

    def test_frustrum_to_harmonics(self):
        R = np.random.rand(self.nb_rays, 6)
        grid = np.random.rand([self.idim, self.idim, self.idim, 10]) # first one for occlusion, between [0,1] the rest for harmonics
        H = frustrum_to_harmonics(R, grid)
        np.testing.assert_array_almost_equal(H.shape, np.array([self.nb_rays, self.idim, self.idim, self.idim, 9]))


if __name__=='__main__':
    unittest.main()