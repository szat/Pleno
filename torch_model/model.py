from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
import torch
import torch.nn as nn
import typing

from frustum_branch_torch import frustum_to_harmonics, rays_to_frustum
from trilinear_interpolation_torch import trilinear_interpolation


class RadianceField(nn.Module):
    def __init__(self,
                 idim: int,
                 nb_samples: int,
                 distr_ray_sampling: Distribution=Uniform,
                 delta_voxel: torch.Tensor=torch.tensor([1, 1, 1], dtype=torch.float)):
        super().__init__()
        self.distr_ray_sampling = distr_ray_sampling
        self.nb_samples = nb_samples
        self.delta_voxel = delta_voxel
        self.idim = idim
        # N x 9 (harmonic coeffs) x 3 (color chs) + 1 (density)
        self.grid = torch.rand((idim, idim, idim, 9), requires_grad=True)
        self.opacity = torch.rand((idim, idim, idim), requires_grad=True)


    def forward(self, x: torch.Tensor, d: torch.Tensor, scale_samples: float):
        """
        x, d define origins and directions of rays
        """
        assert x.shape[0] == d.shape[0]
        sample_obj = self.distr_ray_sampling(torch.zeros(x.shape[0]), torch.ones(x.shape[0]))
        samples, _ = torch.sort(sample_obj.sample_n(self.nb_samples).T * scale_samples) # nb_rays x nb_samples
        frustum, sample_points, dir_vec_neighs = rays_to_frustum(x, d, samples, self.delta_voxel)
        neigh_harmonics, neigh_opacities = frustum_to_harmonics(frustum, dir_vec_neighs, self.grid, self.opacity)

        sample_points = torch.flatten(sample_points, 0, 1)
        neigh_harmonics = torch.flatten(neigh_harmonics, 0, 1)
        neigh_opacities = torch.flatten(neigh_opacities, 0, 1)
        neigh_opacities = neigh_opacities.unsqueeze(2)

        interp_harmonics = trilinear_interpolation(sample_points, neigh_harmonics,
                                                   dx=self.delta_voxel[0], dy=self.delta_voxel[1], dz=self.delta_voxel[2])
        interp_opacities = trilinear_interpolation(sample_points, neigh_opacities,
                                                   dx=self.delta_voxel[0], dy=self.delta_voxel[1], dz=self.delta_voxel[2])

        interp_harmonics = torch.reshape(interp_harmonics, (x.shape[0], self.nb_samples, 9)) # nb_rays x nb_samples x 9
        interp_opacities = torch.reshape(interp_opacities, (x.shape[0], self.nb_samples)) # nb_rays x nb_samples

        # render with interp_harmonics and interp_opacities
