import math

import torch
import unittest

from model import RadianceField


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 100
        # So the grid looks like [0.0, 1.0, ... , 100.0]^3, functions take (dx,dy,dz)=(1,1,1) by default
        self.nb_rays = 7
        self.nb_samples = 11

    def test_rays_to_frustum_shape(self):
        # shifts/scalings are just for testing and making sure that random tesing samples does not overflow grid:
        ray_origins = torch.rand(self.nb_rays, 3) * 20 + self.idim/2
        ray_dir_vecs = torch.rand(self.nb_rays, 3) + 0.5

        model = RadianceField(self.idim, self.nb_samples)
        model(ray_origins, ray_dir_vecs, 30)

        """
        frustum, sample_points, dir_vec_neighs = rays_to_frustrum(ray_origins, ray_dir_vecs, ray_samples)
        torch.testing.assert_close(frustum.shape,
                                   torch.ones(self.nb_rays, self.nb_samples, 8, 3).shape)
        torch.testing.assert_close(dir_vec_neighs.shape,
                                   torch.ones(self.nb_rays, self.nb_samples, 8, 3).shape)
        """

if __name__=='__main__':
    unittest.main()
