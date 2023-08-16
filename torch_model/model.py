from typing import Tuple
import math

from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform
import torch
import torch.nn
import torch.optim

from utils import build_samples, eval_sh_bases, trilinear_interpolation
from torch.profiler import profile, record_function, ProfilerActivity

class RadianceField(torch.nn.Module):

    def __init__(self,
                 idim: int,
                 nb_sh_channels: int, 
                 nb_samples: int,
                 opacity: torch.Tensor = None,
                 grid: torch.Tensor = None,
                 delta_voxel: torch.Tensor=torch.tensor([1, 1, 1], dtype=torch.float),
                 w_tv_harms: float = 1,
                 w_tv_opacity: float = 1,
                 device: str = None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(self.device)
        ## From the svox paper
        self.SH_C0 = 0.28209479177387814
        self.SH_C1 = 0.4886025119029199
        self.SH_C2 = torch.Tensor([
                                    1.0925484305920792,
                                    -1.0925484305920792,
                                    0.31539156525252005,
                                    -1.0925484305920792,
                                    0.5462742152960396
                                  ])
        self.SH_C3 = torch.Tensor([
                                   -0.5900435899266435,
                                   2.890611442640554,
                                   -0.4570457994644658,
                                   0.3731763325901154,
                                   -0.4570457994644658,
                                   1.445305721320277,
                                   -0.5900435899266435
                                  ])
        self.SH_C4 = torch.Tensor([
                                   2.5033429417967046,
                                   -1.7701307697799304,
                                   0.9461746957575601,
                                   -0.6690465435572892,
                                   0.10578554691520431,
                                   -0.6690465435572892,
                                   0.47308734787878004,
                                   -1.7701307697799304,
                                   0.6258357354491761,
                                  ])
        self.nb_samples = nb_samples
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
            self.opacity = torch.nn.Parameter(torch.rand((idim, idim, idim, 1), device=self.device))
        else:
            self.opacity = torch.nn.Parameter(opacity.to(self.device).unsqueeze(3))
        self.box_min = torch.Tensor([[0, 0, 0]]).to(self.device)
        self.inf = torch.prod(idim*self.delta_voxel)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-6)

    def forward(self, x: torch.Tensor, d: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor, get_samples=False) -> torch.Tensor:
        """
        x, d define origins and directions of rays
        """
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        #     with record_function("total_forward"):
        nb_rays = x.shape[0]
        samples = torch.arange(start=0.05, end=0.95, step=1/self.nb_samples, device=self.device, dtype=x.dtype)
        samples = samples.unsqueeze(0).expand(nb_rays, samples.shape[0])
        tmin = tmin.reshape((-1, 1))
        tmax = tmax.reshape((-1, 1))
        samples = (tmax - tmin) * samples + tmin
        sample_points = build_samples(x, d, samples)
        extra = sample_points


                # evaluations harmonics are done at each ray direction:
                # with record_function("sh_part"):
        sh = eval_sh_bases(d, self.SH_C0, self.SH_C1, self.SH_C2, self.SH_C3, self.SH_C4)
        sh = sh.repeat(1, self.nb_sh_channels) # nb_rays x nb_channels*nb_sh
        sample_points = torch.flatten(sample_points, 0, 1)
        interp_sh_coeffs = trilinear_interpolation(sample_points, self.grid, self.box_min, self.delta_voxel)
        interp_opacities = trilinear_interpolation(sample_points, self.opacity, self.box_min, self.delta_voxel)
        interp_opacities = torch.clamp(interp_opacities, 0, 100000)

        # nb_rays x nb_samples x nb_channels*num_sh
        # with record_function("addition_part"):
        interp_sh_coeffs = interp_sh_coeffs.reshape((nb_rays, samples.shape[1], 9*self.nb_sh_channels))
        interp_opacities = interp_opacities.reshape((nb_rays, samples.shape[1])) # nb_rays x nb_samples
        interp_harmonics = interp_sh_coeffs * sh.unsqueeze(1)
        interp_harmonics = torch.reshape(interp_harmonics, (nb_rays, samples.shape[1], self.nb_sh_channels, 9))

        # render with interp_harmonics and interp_opacities:
        deltas = samples[:, 1:] - samples[:, :-1]
        deltas_times_sigmas = deltas * interp_opacities[:, :-1]
        deltas_times_sigmas = -deltas_times_sigmas
        cum_weighted_deltas = torch.cumsum(deltas_times_sigmas, dim=1)
        cum_weighted_deltas = torch.cat([torch.zeros((nb_rays, 1), device=self.device), cum_weighted_deltas[:, :-1]], dim=1)
        deltas_times_sigmas = deltas_times_sigmas.unsqueeze(2)
        cum_weighted_deltas = cum_weighted_deltas.unsqueeze(2)
        samples_color = torch.clamp(torch.sum(interp_harmonics, dim=3) + 0.5, 0.0, 100000)
        rays_color = torch.sum(torch.exp(cum_weighted_deltas) *
                               (1 - torch.exp(deltas_times_sigmas)) * samples_color[:, :-1, :],
                               dim=1)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        if get_samples:
            return rays_color, extra
        else:
            return rays_color

    def render_rays(self, ray_origins: torch.Tensor, ray_dirs: torch.Tensor,
                    tmin: torch.Tensor, tmax: torch.Tensor,
                    batch_size: int, get_samples=False) -> torch.Tensor:

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        #     with record_function("total_render"):
        ray_origins = ray_origins.to(self.device)
        ray_dirs = ray_dirs.to(self.device)
        color_batched = []
        samples_batched = []
        sh_batched = []
        interp_sh_coeffs_batched = []
        interp_opacities_batched = []
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
                if get_samples:
                    colors, extra = self(origins_batched, dirs_batched,tmin_batched, tmax_batched, True)
                    colors = colors.cpu()
                    color_batched.append(colors)
                    return torch.cat(color_batched), extra
                else:
                    colors = self(origins_batched, dirs_batched,tmin_batched, tmax_batched)
                    colors = colors.cpu()
                    color_batched.append(colors)
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
