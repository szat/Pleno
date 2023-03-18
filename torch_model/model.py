from typing import Tuple
import math


from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
import torch
import torch.nn as nn

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
        assert nb_samples > 1
        self.delta_voxel = delta_voxel
        self.idim = idim
        self.grid = torch.rand((idim + 1, idim + 1, idim + 1, 9), requires_grad=True)
        self.opacity = torch.rand((idim + 1, idim + 1, idim + 1), requires_grad=True)
        self.inf = torch.tensor(float(idim)*idim*idim)
        self.box_min = torch.Tensor([[0, 0, 0]])
        self.box_max = torch.Tensor([[float(idim), idim, idim]])

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        x, d define origins and directions of rays
        """

        assert x.shape[0] == d.shape[0]
        nb_rays = x.shape[0]

        ray_inv_dirs = 1. / d
        tmin, tmax = self.intersect_ray_aabb(x, ray_inv_dirs, self.box_min.expand(nb_rays, 3),
                                                              self.box_max.expand(nb_rays, 3))
        mask = torch.Tensor(tmin < tmax)
        assert mask.any() # otherwise, empty operations/gradient
        x = x[mask]
        nb_rays = x.shape[0]
        d = d[mask]
        tmin = tmin[mask]
        tmax = tmax[mask]
        sample_obj = self.distr_ray_sampling(tmin, tmax)
        samples, _ = torch.sort(sample_obj.sample(sample_shape=[self.nb_samples]).T) # nb_rays x nb_samples
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

        interp_harmonics = torch.reshape(interp_harmonics, (nb_rays, self.nb_samples, 9)) # nb_rays x nb_samples x 9
        interp_opacities = torch.reshape(interp_opacities, (nb_rays, self.nb_samples)) # nb_rays x nb_samples

        # render with interp_harmonics and interp_opacities
        cumm_opacity = torch.zeros(nb_rays, dtype=torch.float)
        ray_color = torch.zeros(nb_rays, dtype=torch.float)
        for i in range(self.nb_samples - 1):
            delta_i = samples[:, i+1] - samples[:, i]
            transmittance = torch.exp(-cumm_opacity)
            cur_opacity = delta_i * interp_opacities[:, i]
            color_sample = torch.sigmoid(torch.sum(interp_harmonics[:, i], dim=1))
            ray_color += transmittance * (1 - torch.exp(-cur_opacity)) * color_sample
            cumm_opacity += cur_opacity

        return ray_color

    def intersect_ray_aabb(self,
                           ray_origins: torch.Tensor,
                           ray_inv_dirs: torch.Tensor,
                           box_mins: torch.Tensor,
                           box_maxs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        considers the boundary of the volume as NON intersecting, if tmax <= tmin then NO intersection
        source: http://psgraphics.blogspot.com/2016/02/new-simple-ray-box-test-from-andrew.html
        """

        tmin = torch.ones(len(ray_origins)) * (-self.inf)
        tmax = torch.ones(len(ray_origins)) * self.inf
        t0 = (box_mins - ray_origins) * ray_inv_dirs
        t1 = (box_maxs - ray_origins) * ray_inv_dirs
        tsmaller = torch.min(t0, t1)
        tbigger = torch.max(t0, t1)
        tsmaller_max, _ = torch.max(tsmaller, dim=1)
        tbigger_min, _ = torch.min(tbigger, dim=1)
        tmin, _ = torch.max(torch.stack([tmin, tsmaller_max]), dim=0)
        tmax, _ = torch.min(torch.stack([tmax, tbigger_min]), dim=0)
        return tmin, tmax

