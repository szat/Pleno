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
    # Case when only 1 entry to interpolate, want shape [3, nb]
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
    # a000 = tmpX * tmpY * tmpZ
    # a100 = xd * tmpY * tmpZ
    # a010 = tmpX * yd * tmpZ
    # a110 = xd * yd * tmpZ
    # a001 = tmpX * tmpY * zd
    # a101 = xd * tmpY * zd
    # a011 = tmpX * yd * zd
    # a111 = xd * yd * zd

    a000 = tmpX * tmpY
    a100 = xd * tmpY
    a010 = tmpX * yd
    a110 = xd * yd

    # weights = torch.stack([a000, a001, a010, a011, a100, a101, a110, a111]).unsqueeze(2)
    # weights = torch.stack([a000, a001, a010, a011, a100, a101, a110, a111]).unsqueeze(2)
    # coeff = torch.stack([values[x0, y0, z0], values[x0, y0, z0 + 1],
    #                      values[x0, y0 + 1, z0], values[x0, y0 + 1, z0 + 1],
    #                      values[x0 + 1, y0, z0], values[x0 + 1, y0, z0 + 1],
    #                      values[x0 + 1, y0 + 1, z0], values[x0 + 1, y0 + 1, z0 + 1]])

    weights = torch.stack([a000, a010, a100, a110]).unsqueeze(2)
    coeff = torch.stack([values[x0, y0, z0], values[x0, y0, z0 + 1],
                         values[x0, y0 + 1, z0], values[x0, y0 + 1, z0 + 1],
                         values[x0 + 1, y0, z0], values[x0 + 1, y0, z0 + 1],
                         values[x0 + 1, y0 + 1, z0], values[x0 + 1, y0 + 1, z0 + 1]])

    tmpZ = tmpZ[None, :, None]
    zd = zd[None, :, None]

    new = torch.sum(weights * coeff[[0, 2, 4, 6]], dim=0) * tmpZ \
          + torch.sum(weights * coeff[[1, 3, 5, 7]], dim=0) * zd

    # tmpZ = tmpZ[None, :, None]
    # zd = zd[None, :, None]

    # weights_tmpZ = torch.stack([a000, a001, a010, a011]).unsqueeze(2)
    # weights_zd = torch.stack([a100, a101, a110, a111]).unsqueeze(2)
    # coeff_tmpZ = torch.stack([values[x0, y0, z0], values[x0, y0, z0 + 1],
    #                           values[x0, y0 + 1, z0], values[x0, y0 + 1, z0 + 1]])
    # coeff_zd = torch.stack([values[x0 + 1, y0, z0], values[x0 + 1, y0, z0 + 1],
    #                         values[x0 + 1, y0 + 1, z0], values[x0 + 1, y0 + 1, z0 + 1]])
    old = torch.sum(weights * coeff, dim=0)
    torch.sum(torch.abs(old - new))
    return torch.sum(weights * coeff, dim=0)
