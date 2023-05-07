from typing import Tuple

import numpy as np
import torch

from sampling_branch import intersect_ray_aabb 


def build_samples(ray_origins: torch.Tensor,
                  ray_dir_vecs: torch.Tensor,
                  samples: torch.Tensor,
                 ) -> Tuple[torch.Tensor]:

    n_rays = ray_origins.shape[0]

    # normalize ray_dir_vecs
    r = torch.linalg.norm(ray_dir_vecs, dim=1)
    r = torch.unsqueeze(r, 1)
    ray_dir_vecs = ray_dir_vecs / r

    samples = torch.unsqueeze(samples, dim=2)
    ray_origins = torch.unsqueeze(ray_origins, dim=1)
    ray_dir_vecs = torch.unsqueeze(ray_dir_vecs, dim=1)
    sample_vecs = ray_origins + samples * ray_dir_vecs

    return sample_vecs


def validate_and_find_ray_intersecs(rays_dirs: np.ndarray, rays_origins: np.ndarray):
    """
    finds rays intersections with model cube and choose valid ones
    """

    box_min = -np.ones(rays_origins.shape) * 0.99
    box_max = np.ones(rays_origins.shape) * 0.99
    ray_inv_dirs = 1. / rays_dirs
    tmin, tmax = intersect_ray_aabb(rays_origins, ray_inv_dirs, box_min, box_max)
    mask = tmin < tmax
    valid_rays_origins = torch.from_numpy(rays_origins[mask])
    valid_rays_dirs = torch.from_numpy(rays_dirs[mask])
    valid_tmin = torch.from_numpy(tmin[mask])
    valid_tmax = torch.from_numpy(tmax[mask])
    return valid_rays_origins, valid_rays_dirs, valid_tmin, valid_tmax
        


def eval_sh_bases(dirs: torch.Tensor,
                  SH_C0: float, SH_C1: float,
                  SH_C2: torch.Tensor, SH_C3: torch.Tensor, SH_C4: torch.Tensor,
                  basis_dim: int = 9) -> torch.Tensor:
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    """
    result = torch.empty([dirs.shape[0], basis_dim], dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

    return result


def trilinear_interpolation(vecs: torch.Tensor,
                            values: torch.Tensor,
                            origin: torch.Tensor,
                            delta_voxel: torch.Tensor):
    # Normalize, transform into origin = (0,0,0) and dx = dy = dz = 1
    xyz = vecs - origin
    xyz = xyz / delta_voxel

    xyz_floor = torch.floor(xyz)
    diff = xyz - xyz_floor

    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = xyz_floor.to(torch.long)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
    tmpX = 1 - xd
    tmpY = 1 - yd
    tmpZ = 1 - zd
    a000 = tmpX * tmpY
    a100 = xd * tmpY
    a010 = tmpX * yd
    a110 = xd * yd

    weights = torch.stack([a000, a010, a100, a110]).unsqueeze(2)
    coeff = torch.stack([values[x0, y0, z0], values[x0, y0, z0 + 1],
                         values[x0, y0 + 1, z0], values[x0, y0 + 1, z0 + 1],
                         values[x0 + 1, y0, z0], values[x0 + 1, y0, z0 + 1],
                         values[x0 + 1, y0 + 1, z0], values[x0 + 1, y0 + 1, z0 + 1]])

    tmpZ = tmpZ.unsqueeze(0).unsqueeze(-1)
    zd = zd.unsqueeze(0).unsqueeze(-1)

    inter_values = torch.sum(weights * coeff[[0, 2, 4, 6]], dim=0) * tmpZ + \
                   torch.sum(weights * coeff[[1, 3, 5, 7]], dim=0) * zd
    return inter_values


def trilinear_interpolation_shuffle(vecs: torch.Tensor,
                                    links: torch.Tensor,
                                    values_compressed: torch.Tensor,
                                    origin: torch.Tensor,
                                    delta_voxel: torch.Tensor):

    xyz = vecs - origin
    xyz = xyz / delta_voxel

    xyz_floor = torch.floor(xyz)
    diff = xyz - xyz_floor

    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = xyz_floor.to(torch.long)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
    
    l000 = links[x0, y0, z0]
    l100 = links[x0+1, y0, z0]
    l010 = links[x0, y0+1, z0]
    l001 = links[x0, y0, z0+1]
    l110 = links[x0+1, y0+1, z0]
    l011 = links[x0, y0+1, z0+1]
    l101 = links[x0+1, y0, z0+1]
    l111 = links[x0+1, y0+1, z0+1]

    # These are not all the same masks, since sometimes we are on the boundary
    mask_l000 = l000 >= 0
    mask_l100 = l100 >= 0
    mask_l010 = l010 >= 0
    mask_l001 = l001 >= 0
    mask_l110 = l110 >= 0
    mask_l011 = l011 >= 0
    mask_l101 = l101 >= 0
    mask_l111 = l111 >= 0

    # The or operator gives the boundary
    mask = mask_l000 | mask_l100 | mask_l010 | mask_l001 | mask_l110 | mask_l011 | mask_l101 | mask_l111

    l000[mask & ~mask_l000] = 0
    l100[mask & ~mask_l100] = 0
    l010[mask & ~mask_l010] = 0
    l001[mask & ~mask_l001] = 0
    l110[mask & ~mask_l110] = 0
    l011[mask & ~mask_l011] = 0
    l101[mask & ~mask_l101] = 0
    l111[mask & ~mask_l111] = 0

    l000 = l000[mask]
    l100 = l100[mask]
    l010 = l010[mask]
    l001 = l001[mask]
    l110 = l110[mask]
    l011 = l011[mask]
    l101 = l101[mask]
    l111 = l111[mask]

    v000 = values_compressed[l000]
    v100 = values_compressed[l100]
    v010 = values_compressed[l010]
    v001 = values_compressed[l001]
    v110 = values_compressed[l110]
    v011 = values_compressed[l011]
    v101 = values_compressed[l101]
    v111 = values_compressed[l111]

    xd = xd[mask]
    yd = yd[mask]
    zd = zd[mask]

    tmpX = 1 - xd
    tmpY = 1 - yd
    tmpZ = 1 - zd
    a000 = tmpX * tmpY
    a100 = xd * tmpY
    a010 = tmpX * yd
    a110 = xd * yd

    weights = torch.stack([a000, a010, a100, a110]).unsqueeze(2)
    coeff = torch.stack([v000, v001, v010, v011, v100, v101, v110, v111])
    tmpZ = tmpZ.unsqueeze(0).unsqueeze(-1)
    zd = zd.unsqueeze(0).unsqueeze(-1)

    inter_values = torch.sum(weights * coeff[[0, 2, 4, 6]], dim=0) * tmpZ + \
                   torch.sum(weights * coeff[[1, 3, 5, 7]], dim=0) * zd

    res = torch.zeros((vecs.shape[0], values_compressed.shape[1]),
                      dtype=torch.float64,  
                      device=values_compressed.device)
    res[mask] = inter_values
    return res
