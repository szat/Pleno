import math

import torch
import unittest

from frustum_branch_torch import rays_to_frustum, frustum_to_harmonics
from spherical_harmonics_torch import sh_cartesian


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 100
        # So the grid looks like [0.0, 1.0, ... , 100.0]^3, functions take (dx,dy,dz)=(1,1,1) by default
        self.nb_rays = 7
        self.nb_samples = 11
        # 9 harmonic coeficients per voxel
        self.grid = torch.rand(self.idim, self.idim, self.idim, 9)
        self.opacity = torch.rand(self.idim, self.idim, self.idim)

    def test_rays_to_frustum_shape(self):
        # shifts/scalings are just for testing and making sure that random tesing samples does not overflow grid:
        ray_origins = torch.rand(self.nb_rays, 3) * 20 + self.idim/2
        ray_dir_vecs = torch.rand(self.nb_rays, 3) + 0.5
        ray_samples = torch.rand(self.nb_rays, self.nb_samples) * 30
        frustum, sample_points, dir_vec_neighs = rays_to_frustum(ray_origins, ray_dir_vecs, ray_samples)
        torch.testing.assert_close(frustum.shape,
                                   torch.ones(self.nb_rays, self.nb_samples, 8, 3).shape)
        torch.testing.assert_close(dir_vec_neighs.shape,
                                   torch.ones(self.nb_rays, self.nb_samples, 8, 3).shape)

    def test_frustum_to_harmonics_shape(self):
        # shifts/scalings are just for testing and making sure that random tesing samples does not overflow grid:
        ray_origins = torch.rand(self.nb_rays, 3) * 20 + self.idim/2
        ray_dir_vecs = torch.rand(self.nb_rays, 3) + 0.5
        ray_samples = torch.rand(self.nb_rays, self.nb_samples) * 30
        frustum, sample_points, dir_vec_neighs = rays_to_frustum(ray_origins, ray_dir_vecs, ray_samples)
        neighs_harmonics, neighs_opacities = frustum_to_harmonics(frustum, dir_vec_neighs, self.grid, self.opacity)
        torch.testing.assert_close(neighs_harmonics.shape,
                                   torch.zeros(self.nb_rays, self.nb_samples, 8, 9).shape)
        torch.testing.assert_close(neighs_opacities.shape,
                                   torch.zeros(self.nb_rays, self.nb_samples, 8).shape)

    def test_rays_to_frustum_easy_values(self):
        nr_rays = 4
        nr_samples = 3
        root3 = math.sqrt(3)
        ray_origins = torch.ones((nr_rays, 3)) * 50
        ray_dir_vecs = torch.Tensor([[1, 0, 0], # pointing to +x
                                    [0, 1, 0], # pointing to +y
                                    [0, 0, 1], # pointing to +z
                                    [1, 1, 1]] # pointing to (1,1,1)
                                  ).reshape((nr_rays, 3))
        # samples defined by following distances from ray origin along ray dir:
        ray_samples = torch.stack([torch.Tensor([2.5, 5.7, 8.5]),
                                   torch.Tensor([3.3, 4.9, 9.2]),
                                   torch.Tensor([3.5, 5.7, 7.5]),
                                   torch.Tensor([1.5*root3, 2.2*root3, 5.8*root3])]
                                 ).reshape((nr_rays, nr_samples))

        # test only first neighbour (relative (0, 0, 0)-neighbour)
        voxels_ijk = torch.Tensor([
                                   [[52, 50, 50], [55, 50, 50], [58, 50, 50]],
                                   [[50, 53, 50], [50, 54, 50], [50, 59, 50]],
                                   [[50, 50, 53], [50, 50, 55], [50, 50, 57]],
                                   [[51, 51, 51], [52, 52, 52], [55, 55, 55]]
                                 ])
        dir_vecs = torch.Tensor([
                                 [[2, 0, 0], [5, 0, 0], [8, 0, 0]],
                                 [[0, 3, 0], [0, 4, 0], [0, 9, 0]],
                                 [[0, 0, 3], [0, 0, 5], [0, 0, 7]],
                                 [[1, 1, 1], [2, 2, 2], [5, 5, 5]]
                               ])

        assert voxels_ijk.size() == (nr_rays, nr_samples, 3)
        frustum, sample_points, dir_vec_neighs = rays_to_frustum(ray_origins, ray_dir_vecs, ray_samples)
        torch.testing.assert_close(frustum[:, :, 0, :], voxels_ijk)
        torch.testing.assert_close(dir_vec_neighs[:, :, 0, :], dir_vecs)

    def test_frustum_to_harmonics_easy_values(self):
        nr_rays = 1
        nr_samples = 1
        ray_origins = torch.Tensor([50.5, 50.5, 50.5]).reshape((nr_rays, 3))
        ray_dir_vecs = torch.Tensor([1, 0, 0]).reshape((nr_rays, 3)) # pointing to +x
        ray_samples = torch.Tensor([0]).reshape((nr_rays, nr_samples)) # sample at ray origin
        frustum, sample_points, dir_vec_neighs = rays_to_frustum(ray_origins, ray_dir_vecs, ray_samples)
        ret_harmonics, ret_opacities = frustum_to_harmonics(frustum, dir_vec_neighs, self.grid, self.opacity)

        true_opacities = torch.Tensor([self.opacity[50, 50, 50], self.opacity[51, 50, 50],
                                       self.opacity[50, 51, 50], self.opacity[51, 51, 50],
                                       self.opacity[50, 50, 51], self.opacity[51, 50, 51],
                                       self.opacity[50, 51, 51], self.opacity[51, 51, 51]])
        true_opacities = true_opacities.reshape((nr_rays, nr_samples, 8))
        assert ret_opacities.shape == true_opacities.shape
        torch.testing.assert_close(ret_opacities, true_opacities)

        coeff_harmonics = torch.stack([self.grid[50, 50, 50], self.grid[51, 50, 50],
                                       self.grid[50, 51, 50], self.grid[51, 51, 50],
                                       self.grid[50, 50, 51], self.grid[51, 50, 51],
                                       self.grid[50, 51, 51], self.grid[51, 51, 51]])
        coeff_harmonics = torch.unsqueeze(coeff_harmonics, axis=0)
        coeff_harmonics = torch.unsqueeze(coeff_harmonics, axis=1) # 1 x 1 x 8 x 9
        true_harmonics = sh_cartesian(torch.Tensor([[-1, -1, -1], [1, -1, -1],
                                                    [-1, 1, -1], [1, 1, -1],
                                                    [-1, -1, 1], [1, -1, 1],
                                                    [-1, 1, 1], [1, 1, 1]]
                                                  ).reshape((8, 3)))
        true_weighted_harmonics = coeff_harmonics * true_harmonics.reshape((1, 1, 8, 9))
        assert ret_harmonics.shape == true_weighted_harmonics.shape
        torch.testing.assert_close(ret_harmonics, true_weighted_harmonics)


if __name__=='__main__':
    unittest.main()
