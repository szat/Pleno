import torch

def trilinear_interpolation(xyz, c, origin=torch.zeros(3), dx=1.0, dy=1.0, dz=1.0):
    # Normalize, transform into origin = (0,0,0) and dx = dy = dz = 1
    # Case when only 1 entry to interpolate, want shape [3, nb]
    if len(xyz.size()) == 1:
        xyz = xyz.unsqueeze(1)
        xyz = xyz.T

    xyz = xyz - origin
    xyz = xyz / torch.tensor([dx, dy, dz]) #, dtype=float), axis=1)

    xyz_floor = torch.floor(xyz)
    diff = xyz - xyz_floor

    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = xyz_floor.to(torch.long)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
    tmp = 1 - xd

    x_zeros = torch.zeros(x0.size(), dtype=torch.long)
    x_zeros[xd > 0] +=1
    x1 = x0 + x_zeros

    y_zeros = torch.zeros(y0.size(), dtype=torch.long)
    y_zeros[yd > 0] +=1
    y1 = y0 + y_zeros

    z_zeros = torch.zeros(z0.size(), dtype=torch.long)
    z_zeros[zd > 0] +=1
    z1 = z0 + z_zeros

    c00 = c[x0, y0, z0] * tmp + c[x1, y0, z0] * xd
    c01 = c[x0, y1, z0] * tmp + c[x1, y1, z0] * xd
    c10 = c[x0, y0, z1] * tmp + c[x1, y0, z1] * xd
    c11 = c[x0, y1, z1] * tmp + c[x1, y1, z1] * xd

    tmp = 1 - yd
    c0 = c00 * tmp + c10 * yd
    c1 = c01 * tmp + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    if len(c.size()) == 1:
        c = c.unsqueeze(1)
    return c
