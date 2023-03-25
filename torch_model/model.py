from typing import Tuple
import math

from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
import torch
import torch.nn
import torch.optim

from frustum_branch_torch import frustum_to_harmonics, rays_to_frustum
from trilinear_interpolation_torch import trilinear_interpolation


class RadianceField(torch.nn.Module):

    def __init__(self,
                 idim: int,
                 nb_samples: int,
                 distr_ray_sampling: Distribution=Uniform,
                 delta_voxel: torch.Tensor=torch.tensor([1, 1, 1], dtype=torch.float),
                 w_tv_harms: float = 1,
                 w_tv_opacity: float = 1):
        super().__init__()
        self.distr_ray_sampling = distr_ray_sampling
        self.nb_samples = nb_samples
        assert nb_samples > 1
        self.delta_voxel = delta_voxel
        self.idim = idim
        self.w_tv_harms = w_tv_harms
        self.w_tv_opacity = w_tv_opacity
        # self.grid = torch.nn.Parameter(torch.rand((idim + 1, idim + 1, idim + 1, 9)))
        # self.opacity = torch.nn.Parameter(torch.rand((idim + 1, idim + 1, idim + 1)))
        self.grid = torch.nn.Parameter(torch.rand((idim, idim, idim, 9)))
        self.opacity = torch.nn.Parameter(torch.rand((idim, idim, idim)))
        # self.grid = torch.nn.Parameter(torch.FloatTensor(idim, idim, idim, 9)).to('cuda')
        # self.opacity = torch.nn.Parameter(torch.FloatTensor(idim, idim, idim)).to('cuda')
        self.inf = torch.tensor(float(idim)*idim*idim)
        self.box_min = torch.Tensor([[0, 0, 0]])
        # self.box_max = torch.Tensor([[float(idim), idim, idim]])
        self.box_max = torch.Tensor([[float(idim-1), idim-1, idim-1]])
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-6)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
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
        sample_obj = self.distr_ray_sampling(tmin, tmax) # could use custom distr according to dendity e.g.
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

        # render with interp_harmonics and interp_opacities:
        cumm_opacities = torch.zeros(nb_rays, dtype=torch.float)
        rays_color = torch.zeros(nb_rays, dtype=torch.float)
        for i in range(self.nb_samples - 1):
            deltas_i = samples[:, i+1] - samples[:, i]
            transmittances = torch.exp(-cumm_opacities)
            cur_opacities = deltas_i * interp_opacities[:, i]
            samples_color = torch.sigmoid(torch.sum(interp_harmonics[:, i], dim=1))
            rays_color += transmittances * (1 - torch.exp(-cur_opacities)) * samples_color
            cumm_opacities += cur_opacities

        return rays_color

    def total_variation(self, voxels_ijk_tv: torch.Tensor) -> Tuple[float, float]:

        index_delta_x = voxels_ijk_tv[:, 0] + 1
        index_delta_y = voxels_ijk_tv[:, 1] + 1
        index_delta_z = voxels_ijk_tv[:, 2] + 1

        # ignore voxels in the boundary:
        voxels_ijk_tv = voxels_ijk_tv[index_delta_x <= self.idim]
        voxels_ijk_tv = voxels_ijk_tv[index_delta_y <= self.idim]
        voxels_ijk_tv = voxels_ijk_tv[index_delta_z <= self.idim]
        assert voxels_ijk_tv.shape[0] > 0
        # create delta voxel indexes per dimension:
        index_delta_x = torch.stack([voxels_ijk_tv[:, 0] + 1, voxels_ijk_tv[:, 1], voxels_ijk_tv[:, 2]])
        index_delta_y = torch.stack([voxels_ijk_tv[:, 0], voxels_ijk_tv[:, 1] + 1, voxels_ijk_tv[:, 2]])
        index_delta_z = torch.stack([voxels_ijk_tv[:, 0], voxels_ijk_tv[:, 1], voxels_ijk_tv[:, 2] + 1])
        
        delta_sqr_x_harm = self.grid[tuple(index_delta_x.long())] - \
                           self.grid[tuple(voxels_ijk_tv.permute((1, 0)).long())] 
        delta_sqr_x_harm = delta_sqr_x_harm.permute((1, 0))  # nb_voxes x 9
        delta_sqr_x_harm = torch.sum(torch.square(delta_sqr_x_harm), dim=1) / (256 / self.idim)

        delta_sqr_y_harm = self.grid[tuple(index_delta_y.long())] - \
                           self.grid[tuple(voxels_ijk_tv.permute((1, 0)).long())] 
        delta_sqr_y_harm = delta_sqr_y_harm.permute((1, 0))  # nb_voxes x 9
        delta_sqr_y_harm = torch.sum(torch.square(delta_sqr_y_harm), dim=1) / (256 / self.idim)

        delta_sqr_z_harm = self.grid[tuple(index_delta_z.long())] - \
                           self.grid[tuple(voxels_ijk_tv.permute((1, 0)).long())] 
        delta_sqr_z_harm = delta_sqr_z_harm.permute((1, 0))  # nb_voxes x 9
        delta_sqr_z_harm = torch.sum(torch.square(delta_sqr_z_harm), dim=1) / (256 / self.idim)

        delta_root_harm = torch.sqrt(delta_sqr_x_harm + delta_sqr_y_harm + delta_sqr_z_harm)

        delta_x_opac = self.opacity[tuple(index_delta_x.long())] - \
                       self.opacity[tuple(voxels_ijk_tv.permute((1, 0)).long())]

        delta_y_opac = self.opacity[tuple(index_delta_y.long())] - \
                       self.opacity[tuple(voxels_ijk_tv.permute((1, 0)).long())]

        delta_z_opac = self.opacity[tuple(index_delta_z.long())] - \
                       self.opacity[tuple(voxels_ijk_tv.permute((1, 0)).long())]

        delta_root_opac = torch.sqrt((delta_x_opac * delta_x_opac + delta_y_opac * delta_y_opac + \
                                     delta_z_opac * delta_z_opac) / (256 / self.idim))

        return torch.sum(delta_root_harm) / voxels_ijk_tv.shape[0], \
               torch.sum(delta_root_opac) / voxels_ijk_tv.shape[0]

    def train_step(self,
                   train_origins: torch.Tensor,
                   train_dirs: torch.Tensor,
                   train_colors: torch.Tensor,
                   voxels_ijk_tv: torch.Tensor):

        batch_size = train_origins.shape[0]
        pred_colors = self.forward(train_colors, train_dirs)
        assert batch_size == pred_colors.shape[0]
        tv_harmonics, tv_opacity = self.total_variation(voxels_ijk_tv)
        loss = self.criterion(pred_colors, train_colors) + self.w_tv_harms * tv_harmonics + self.w_tv_opacity * tv_opacity
        self.optimizer.zero_grad() # reset grad
        loss.backward() # back propagate
        self.optimizer.step() # update weights

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

