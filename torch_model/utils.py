from typing import Tuple
import torch


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


def sh_cartesian(xyz: torch.Tensor, K: torch.Tensor):
    if xyz.ndim == 1:
        r = torch.linalg.norm(xyz)
        xyz = xyz / r
        x, y, z = xyz[0], xyz[1], xyz[2]
        vec = torch.Tensor([1, y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2],
                           device=xyz.device)
        return vec * K
    else:
        r = torch.linalg.norm(xyz, axis=1)
        r = torch.unsqueeze(r, 1)
        xyz = xyz / r
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ones = torch.ones(x.size(), device=xyz.device)
        vec = torch.vstack([ones, y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2]).T
        return vec * K


def trilinear_interpolation(vecs: torch.Tensor,
                            values: torch.Tensor,
                            origin: torch.Tensor,
                            delta_voxel: torch.Tensor):
    # Normalize, transform into origin = (0,0,0) and dx = dy = dz = 1
    # Case when only 1 entry to interpolate, want shape [3, nb]
    xyz = vecs - origin
    xyz = xyz / delta_voxel

    xyz_floor = torch.floor(xyz)
    diff = xyz - xyz_floor

    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = xyz_floor.to(torch.long)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
    tmp = 1 - xd

    xd = xd.unsqueeze(1)
    yd = yd.unsqueeze(1)
    zd = zd.unsqueeze(1)
    tmp = tmp.unsqueeze(1)
    c00 = values[x0, y0, z0] * tmp + values[x0 + 1, y0, z0] * xd
    c01 = values[x0, y0 + 1, z0] * tmp + values[x0 + 1, y0 + 1, z0] * xd
    c10 = values[x0, y0, z0 + 1] * tmp + values[x0 + 1, y0, z0 + 1] * xd
    c11 = values[x0, y0 + 1, z0 + 1] * tmp + values[x0 + 1, y0 + 1, z0 + 1] * xd

    tmp = 1 - yd
    c0 = c00 * tmp + c10 * yd
    c1 = c01 * tmp + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c
