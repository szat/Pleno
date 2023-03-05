import numpy as np
import unittest
from scipy.special import *
from frustum_branch import *


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 100
        # So the grid looks like [0.0, 1.0, ... , 100.0]^3, functions take (dx,dy,dz)=(1,1,1) by default
        self.nb_rays = 7
        self.nb_samples = 11
        # 9 harmonic coeficients per color channel, 3 channels
        self.grid = np.random.rand(self.idim, self.idim, self.idim, 3, 9)
        self.opacity = np.random.rand(self.idim, self.idim, self.idim)

    def test_rays_to_frustrum_shape(self):
        # shifts/scalings are just for testing and making sure that random tesing samples does not overflow grid:
        ray_origins = np.random.rand(self.nb_rays, 3) * 20 + self.idim/2
        ray_dirs = np.random.rand(self.nb_rays, 2) * np.array([np.pi, 2*np.pi]) # (theta, phi)
        ray_samples = np.random.rand(self.nb_rays, self.nb_samples) * 30
        frustum = rays_to_frustrum(ray_origins, ray_dirs, ray_samples)
        np.testing.assert_array_almost_equal(frustum.shape,
                                             np.array([self.nb_rays, self.nb_samples, 8, 3]))

    def test_frustrum_to_harmonics_shape(self):
        # shifts/scalings are just for testing and making sure that random tesing samples does not overflow grid:
        ray_origins = np.random.rand(self.nb_rays, 3) * 20 + self.idim/2
        ray_dirs = np.random.rand(self.nb_rays, 2) * np.array([np.pi, 2*np.pi]) # (theta, phi)
        ray_samples = np.random.rand(self.nb_rays, self.nb_samples) * 30
        frustum = rays_to_frustrum(ray_origins, ray_dirs, ray_samples)
        neighs_harmonics, neighs_opacities = frustrum_to_harmonics(frustum, ray_dirs, self.grid, self.opacity)
        np.testing.assert_array_almost_equal(neighs_harmonics.shape,
                                             np.array([self.nb_rays, self.nb_samples, 8, 3, 9]))
        np.testing.assert_array_almost_equal(neighs_opacities.shape,
                                             np.array([self.nb_rays, self.nb_samples, 8]))

    def test_rays_to_frustum_easy_values(self):
        nr_rays = 4
        nr_samples = 3
        root3 = np.sqrt(3)
        ray_origins = np.ones((nr_rays, 3)) * 50
        ray_dirs = np.array([[np.pi/2, 0], # pointing to +x
                             [np.pi/2, np.pi/2], # pointing to +y
                             [0, 0], # pointing to +z
                             [0.95531661812451, np.pi/4]] # pointing to (1,1,1)
                           ).reshape((nr_rays, 2))
        # samples defined by following distances from ray origin along ray dir:
        ray_samples = np.stack([np.array([2.5, 5.7, 8.5]),
                                np.array([3.3, 4.9, 9.2]),
                                np.array([3.5, 5.7, 7.5]),
                                np.array([1.5*root3, 2.2*root3, 5.8*root3])]
                              ).reshape((nr_rays, nr_samples))

        # test only first neighbour (relative (0, 0, 0)-neighbour)
        voxels_ijk = np.array([
                                [[52, 50, 50], [55, 50, 50], [58, 50, 50]],
                                [[50, 53, 50], [50, 54, 50], [50, 59, 50]],
                                [[50, 50, 53], [50, 50, 55], [50, 50, 57]],
                                [[51, 51, 51], [52, 52, 52], [55, 55, 55]]
                              ])

        assert voxels_ijk.shape == (nr_rays, nr_samples, 3)
        frustum = rays_to_frustrum(ray_origins, ray_dirs, ray_samples)
        np.testing.assert_array_almost_equal(frustum[:, :, 0, :],
                                             voxels_ijk)


if __name__=='__main__':
    unittest.main()
