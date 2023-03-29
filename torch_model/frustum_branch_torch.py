from typing import Tuple
import torch


def rays_to_frustum(ray_origins: torch.Tensor, ray_dir_vecs: torch.Tensor, samples: torch.Tensor,
                    delta_ijk: torch.Tensor,
                    delta_voxel: torch.Tensor = torch.tensor([1, 1, 1], dtype=torch.float)
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the bottom-left-closest voxel (integer coordinates) of for the given rays and samples.

    A ray is defined by its origin (x0, y0, z0) and its direction vector (xd, yd, zd)

    A sample along a ray is defined by its distance s to the ray origin:
           (x0, y0, z0) + s * (xd, yd, zd)

    Args: 
        ray_origins (torch.Tensor): of shape (nb_rays, 3), second dim gives xyz
        ray_dir_vecs (torch.Tensor): of shape (nb_rays, 3), second dim gives (xd,yd,zd)
        samples (torch.Tensor): of shape (nb_rays, nb_samples), second dim gives the distance samples
                 for the ray as defined above
        delta_voxel (torch.Tensor): of length 3, defining (dx, dy, dz) for voxels. Default (1, 1, 1)

    Returns: Tuple[torch.Tensor, torch.Tensor] first tensor of shape (nb_rays, nb_samples, 8, 3), fourth dim gives the integer 3d-coordinates ijk
    of 8 neightbours of bottom-left-closest voxel containing resp. sample.
    second tensor of shape (nb_rays, nb_samples, 3) gives sampled points along each ray 
    """

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

    return frustum, sample_points


def frustum_to_harmonics(frustum: torch.Tensor, grid: torch.Tensor, opacity: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """computes the weighted multiplication of harmonics (evaluated at the direction to the ray origin) per ray
    and the model coefficients of the 8 integer neighbours defined by frustum per sample.
    
    Args: 
        frustum (torch.Tensor): of shape (nb_rays, nb_samples, 8, 3), given by rays_to_frustrum function
        grid (torch.Tensor): of shape (xdim, ydim zdim, 9) giving model 9 harmonic coeficients
        opacity (torch.Tensor): of shape (xdim, ydim, zdim) giving model opacity
        
    Returns: Tuple[torch.Tensor, torch.tesnor] resp. neigh_harmonics and neigh_densities. First array is of shape
    (nb_rays, nb_samples, 8, 9) and gives the evaluated 9 harmonics weigthed by model coefficients
    per 8 neighbours of each sample of each ray. Second array is of shape (nb_rays, nb_samples, 8) and
    gives the model opacities at all 8 neighbours of each sample of each ray.
    """
    
    
    # retrieve model coefficients of harmonics at 8 neigbour indexes:
    neigh_harmonics_coeff = grid[tuple(torch.permute(frustum, (3, 2, 1, 0)).long())]
    # resulting array of shape nb_rays x nb_samples x 8 x 9:
    neigh_harmonics_coeff = neigh_harmonics_coeff.permute((2, 1, 0, 3))

    # retrieve model density at 8 neigbour indexes:
    neigh_densities = opacity[tuple(frustum.permute((3, 2, 1, 0)).long())]
    neigh_densities = neigh_densities.permute((2, 1, 0))  # nb_rays x nb_samples x 8

    return neigh_harmonics_coeff, neigh_densities
