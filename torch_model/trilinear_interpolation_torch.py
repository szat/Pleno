import torch

def trilinear_interpolation(vecs, c, origin, delta_voxel):
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
    c00 = c[:, 0, :] * tmp + c[:, 1, :] * xd
    c01 = c[:, 2, :] * tmp + c[:, 3, :] * xd
    c10 = c[:, 4, :] * tmp + c[:, 5, :] * xd
    c11 = c[:, 6, :] * tmp + c[:, 7, :] * xd

    tmp = 1 - yd
    c0 = c00 * tmp + c10 * yd
    c1 = c01 * tmp + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c
