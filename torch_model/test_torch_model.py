import math

import numpy as np
import torch
import unittest

from model import RadianceField


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 10
        # So the grid looks like [0.0, 1.0, ... , 100.0]^3, functions take (dx,dy,dz)=(1,1,1) by default
        self.nb_rays = 7
        self.nb_samples = 11

    def test_rays_to_frustum_shape(self):
        thetas = torch.rand(self.nb_rays) *  np.pi
        phis = torch.rand(self.nb_rays) * 2 * np.pi
        ray_dir_vecs = torch.stack([torch.sin(thetas)*torch.cos(phis),
                                torch.sin(thetas)*torch.sin(phis),
                                torch.cos(thetas)]).T
        ray_origins = -30 * ray_dir_vecs + self.idim/2

        model = RadianceField(self.idim, self.nb_samples)
        print(model(ray_origins, ray_dir_vecs))


if __name__=='__main__':
    unittest.main()
