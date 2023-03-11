
from torch.distributions.distribution import Distribution
from torch import Tensor
import torch
import torch.nn as nn
import typing


class RadianceField(nn.Module):
    def __init__(self,
                 N: int,
                 distr_ray_sampling: Distribution,
                 n_ray_sampling: int,
                 resolution: Tuple[int, int, int]):
        super().__init__()
        self.distr_ray_sampling = distr_ray_sampling
        self.n_ray_sampling = n_ray_sampling
        self.resolution = resolution
        self.N = N
        # N x 9 (harmonic coeffs) x 3 (color chs) + 1 (density)
        self.params_table = [torch.zeros(28,
                                         dtype=torch.FloatTensor,
                                         requires_grad=True)
                             for _ in range(N)]



    def forward(self, x: Tensor, d: Tensor):
        # r is a ray
        samples = self.distr_ray_sampling.sample_n(self.n_ray_sampling).sort()
        for sample in samples:
            sampled_point =  x + sample * d

