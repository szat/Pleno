import numpy as np
import unittest
from scipy.special import *
# from renderer import *
from spherical_harmonics import *
from sampling_branch import *


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 5
        self.diff = 1.0
        # So the grid goes like [0.0, 1.0, ... , 4.0]
        self.nb_rays = 7
        self.nb_samples = 11

    def test_rays_to_samples(self):
        R = np.random.rand(self.nb_rays, 6)
        S = rays_to_samples(R, self.nb_samples)
        np.testing.assert_array_almost_equal(S.shape, np.array([self.nb_rays, self.nb_samples, 3]))

    def test_edge_case

    def test_samples_to_icoeffs(self):
        S = np.random.rand(self.nb_rays, self.nb_samples, 3)
        IC = samples_to_icoeffs(S)
        np.testing.assert_array_almost_equal(IC.shape, np.array([self.nb_rays, self.nb_samples, 8]))

    def test_samples_to_neighbors(self):
        S = np.random.rand(self.nb_rays, self.nb_samples, 3)
        N = samples_to_nbh(S)
        np.testing.assert_array_almost_equal(N.shape, np.array([self.nb_rays, self.nb_samples, 8]))




if __name__=='__main__':
    unittest.main()