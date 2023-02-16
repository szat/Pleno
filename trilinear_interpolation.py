import numpy as np


def trilinear_interpolation(xyz, c, origin=np.zeros(3), dx=1.0, dy=1.0, dz=1.0):
    # Normalize, transform into origin = (0,0,0) and dx = dy = dz = 1
    # Case when only 1 entry to interpolate, want shape [3, nb]
    if len(xyz.shape) == 1:
        xyz = np.expand_dims(xyz, axis=1)
        xyz = xyz.T
    # if len(origin.shape) == 1:
    #     origin = np.expand_dims(origin, axis=1)

    xyz = xyz - origin
    xyz = xyz / np.array([dx, dy, dz]) #, dtype=float), axis=1)

    xyz_floor = np.floor(xyz)
    diff = xyz - xyz_floor

    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = xyz_floor.astype(int)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
    tmp = 1 - xd

    c00 = c[x0, y0, z0] * tmp + c[x0 + 1, y0, z0] * xd
    c01 = c[x0, y0 + 1, z0] * tmp + c[x0 + 1, y0 + 1, z0] * xd
    c10 = c[x0, y0, z0 + 1] * tmp + c[x0 + 1, y0, z0 + 1] * xd
    c11 = c[x0, y0 + 1, z0 + 1] * tmp + c[x0 + 1, y0 + 1, z0 + 1] * xd

    tmp = 1 - yd
    c0 = c00 * tmp + c10 * yd
    c1 = c01 * tmp + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    if len(c.shape) == 1:
        c = np.expand_dims(c, axis=1)
    return c
