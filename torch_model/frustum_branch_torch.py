from typing import Tuple
import torch

from spherical_harmonics_torch import sh_cartesian

# relative 8 ijk-neighbours:
delta_ijk = torch.tensor([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                          (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
                         dtype=torch.float).reshape((8, 3))

def rays_to_frustrum(ray_origins: torch.Tensor, ray_dir_vecs: torch.Tensor, samples: torch.Tensor,
                     delta_voxel: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float)
                     ) -> Tuple[torch.Tensor, torch.Tensor]:

    n_rays = ray_origins.shape[0]
    assert n_rays == ray_dir_vecs.shape[0]
    assert n_rays == samples.shape[0]

    # normalize ray_dir_vecs
    r = torch.linalg.norm(ray_dir_vecs, dim=1)
    r = torch.unsqueeze(r, 1)
    assert torch.all(r)
    ray_dir_vecs = ray_dir_vecs / r

    samples = torch.unsqueeze(samples, dim=2)
    ray_origins = torch.unsqueeze(ray_origins, dim=1)
    ray_dir_vecs = torch.unsqueeze(ray_dir_vecs, dim=1)
    sample_points = ray_origins + samples * ray_dir_vecs

    # get ijk coordinates according to (dx, dy, dz):
    frustum = torch.div(sample_points, delta_voxel, rounding_mode='floor').to(torch.int)

    # compute integer index coords of 8 neighbours:
    frustum = torch.unsqueeze(frustum, dim=2)
    frustum = frustum + delta_ijk

    ray_origins = torch.unsqueeze(ray_origins, dim=2)
    dir_vec_neighs = frustum * delta_voxel - ray_origins 

    return frustum, dir_vec_neighs


def frustrum_to_harmonics(frustum: torch.Tensor, dir_vec_neighs: torch.Tensor,
                          grid: torch.Tensor, opacity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
    assert frustum.shape == dir_vec_neighs.shape
    
    # retrieve model coefficients of harmonics at 8 neigbour indexes:
    neigh_harmonics_coeff = grid[tuple(torch.permute(frustum, (3, 2, 1, 0)).long())]
    # resulting array of shape nb_rays x nb_samples x 8 x 9:
    neigh_harmonics_coeff = neigh_harmonics_coeff.permute((2, 1, 0, 3))

    # retrieve model density at 8 neigbour indexes:
    neigh_densities = opacity[tuple(frustum.permute((3, 2, 1, 0)).long())]
    neigh_densities = neigh_densities.permute((2, 1, 0))  # nb_rays x nb_samples x 8

    # evaluations of 8 neighbours harmonics are done in their own direction:
    sh = sh_cartesian(dir_vec_neighs.reshape((-1, 3)))  # (nb_rays * nb_samples * 8) x 9
    sh = sh.reshape((dir_vec_neighs.shape[0], dir_vec_neighs.shape[1], 8, 9))  # nb_rays x nb_samples x 8 x 9
    
    # weigh harmonics with model grid coefficients at all 8 neighbours
    neigh_harmonics = neigh_harmonics_coeff * sh 

    return neigh_harmonics, neigh_densities
