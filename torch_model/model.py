from typing import Tuple
import math

from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
import torch
import torch.nn
import torch.optim

from frustum_branch_torch import frustum_to_harmonics, rays_to_frustum
from spherical_harmonics_torch import sh_cartesian
from trilinear_interpolation_torch import trilinear_interpolation


class RadianceField(torch.nn.Module):

    def __init__(self,
                 idim: int,
                 nb_sh_channels: int, 
                 nb_samples: int,
                 opacity: torch.Tensor = None,
                 grid: torch.Tensor = None,
                 delta_voxel: torch.Tensor=torch.tensor([1, 1, 1], dtype=torch.float),
                 w_tv_harms: float = 1,
                 w_tv_opacity: float = 1):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # relative 8 ijk-neighbours:
        self.delta_ijk = torch.tensor([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                       (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
                                       dtype=torch.int64).reshape((8, 3)).to(self.device)
        self.K_sh = torch.Tensor([0.28209479, 0.48860251, 0.48860251, 0.48860251, 1.09254843,
                                 1.09254843, 0.31539157, 1.09254843, 0.54627422]).to(self.device)
        self.nb_samples = nb_samples
        assert nb_samples > 1
        self.delta_voxel = delta_voxel.to(self.device)
        self.idim = idim
        self.nb_sh_channels = nb_sh_channels
        self.w_tv_harms = w_tv_harms
        self.w_tv_opacity = w_tv_opacity
        if grid is None:
            self.grid = torch.nn.Parameter(torch.rand((idim, idim, idim, 9*nb_sh_channels), device=self.device))
        else:
            self.grid = torch.nn.Parameter(grid.to(self.device))
        if opacity is None:
            self.opacity = torch.nn.Parameter(torch.rand((idim, idim, idim), device=self.device))
        else:
            self.opacity = torch.nn.Parameter(opacity.to(self.device))
        self.inf = torch.tensor(float(idim)*idim*idim)
        self.box_min = torch.Tensor([[0, 0, 0]]).to(self.device)
        self.box_max = torch.Tensor([[float(idim-1), idim-1, idim-1]]).to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-6)

    def forward(self, x: torch.Tensor, d: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
        """
        x, d define origins and directions of rays
        """

        nb_rays = x.shape[0]

        samples = torch.rand((nb_rays, self.nb_samples), dtype=x.dtype, device=self.device)
        tmin = tmin.reshape((-1, 1))
        tmax = tmax.reshape((-1, 1))
        samples = (tmax - tmin) * samples + tmin
        samples, _ = torch.sort(samples, dim=1)
        frustum, sample_points  = rays_to_frustum(x, d, samples, self.delta_ijk, self.delta_voxel)
        neigh_harmonics_coeffs, neigh_opacities = frustum_to_harmonics(frustum, self.grid, self.opacity)

        # evaluations harmonics are done at each ray direction:
        sh = sh_cartesian(d, self.K_sh).repeat(1, self.nb_sh_channels) # nb_rays x nb_channels*nb_sh

        sample_points = torch.flatten(sample_points, 0, 1)
        neigh_harmonics_coeffs = torch.flatten(neigh_harmonics_coeffs, 0, 1) # nb_rays*nb_samples x 8 x nb_channels*nb_sh
        neigh_opacities = torch.flatten(neigh_opacities, 0, 1)
        neigh_opacities = neigh_opacities.unsqueeze(2)

        interp_sh_coeffs = trilinear_interpolation(sample_points, neigh_harmonics_coeffs, self.box_min, self.delta_voxel)
        interp_opacities = trilinear_interpolation(sample_points, neigh_opacities, self.box_min, self.delta_voxel)
        
        # nb_rays x nb_samples x nb_channels*num_sh
        interp_sh_coeffs = torch.reshape(interp_sh_coeffs, (nb_rays, self.nb_samples, 9*self.nb_sh_channels)) 
        interp_opacities = torch.reshape(interp_opacities, (nb_rays, self.nb_samples)) # nb_rays x nb_samples
        interp_harmonics = interp_sh_coeffs * sh.unsqueeze(1)
        interp_harmonics = torch.reshape(interp_harmonics, (nb_rays, self.nb_samples, self.nb_sh_channels, 9))
        
        # render with interp_harmonics and interp_opacities:
        deltas = samples[:, 1:] - samples[:, :-1]
        deltas_times_sigmas = deltas * interp_opacities[:, :-1]
        cum_weighted_deltas = torch.cumsum(deltas_times_sigmas, dim=1)
        cum_weighted_deltas = torch.cat([torch.zeros((nb_rays, 1), device=self.device), cum_weighted_deltas[:, :-1]], dim=1)
        samples_color = torch.clamp_min(torch.sum(interp_harmonics, dim=3) + 0.5, 0.0)
        cum_weighted_deltas = cum_weighted_deltas.unsqueeze(2)
        deltas_times_sigmas = deltas_times_sigmas.unsqueeze(2)
        rays_color = torch.sum(torch.exp(-cum_weighted_deltas) * 
                               (1 - torch.exp(-deltas_times_sigmas)) * samples_color[:, :-1, :],
                               dim=1)
        return rays_color

    def render_rays(self, ray_origins: torch.Tensor, ray_dirs: torch.Tensor,
                    tmin: torch.Tensor, tmax: torch.Tensor,
                    batch_size: int) -> torch.Tensor:

        ray_origins = ray_origins.to(self.device)
        ray_dirs = ray_dirs.to(self.device)
        color_batched = []
        with torch.no_grad():
            for batch_start in range(0, ray_dirs.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, ray_dirs.shape[0])
                origins_batched = ray_origins[batch_start:batch_end]
                origins_batched = origins_batched.to(self.device)
                dirs_batched = ray_dirs[batch_start:batch_end]
                dirs_batched = dirs_batched.to(self.device)
                tmin_batched = tmin[batch_start:batch_end]
                tmin_batched = tmin_batched.to(self.device)
                tmax_batched = tmax[batch_start:batch_end]
                tmax_batched = tmax_batched.to(self.device)
                color_batched.append(self(origins_batched, dirs_batched,
                                          tmin_batched, tmax_batched).cpu())

        return torch.cat(color_batched)

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
