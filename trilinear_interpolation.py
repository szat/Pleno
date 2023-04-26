import numpy as np


def trilinear_interpolation(xyz, color, origin=np.zeros(3), dx=1.0, dy=1.0, dz=1.0):
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

    x_zeros = np.zeros(x0.shape, dtype=int)
    x_zeros[xd >= 0] +=1
    x1 = x0 + x_zeros

    y_zeros = np.zeros(y0.shape, dtype=int)
    y_zeros[yd >= 0] +=1
    y1 = y0 + y_zeros

    z_zeros = np.zeros(z0.shape, dtype=int)
    z_zeros[zd >= 0] +=1
    z1 = z0 + z_zeros

    tmp = np.expand_dims(tmp, 1)
    xd = np.expand_dims(xd, 1)
    yd = np.expand_dims(yd, 1)
    zd = np.expand_dims(zd, 1)

    c00 = color[x0, y0, z0] * tmp + color[x1, y0, z0] * xd
    c01 = color[x0, y1, z0] * tmp + color[x1, y1, z0] * xd
    c10 = color[x0, y0, z1] * tmp + color[x1, y0, z1] * xd
    c11 = color[x0, y1, z1] * tmp + color[x1, y1, z1] * xd

    # c00 = c[x0, y0, z0] * tmp + c[x0 + 1, y0, z0] * xd
    # c01 = c[x0, y0 + 1, z0] * tmp + c[x0 + 1, y0 + 1, z0] * xd
    # c10 = c[x0, y0, z0 + 1] * tmp + c[x0 + 1, y0, z0 + 1] * xd
    # c11 = c[x0, y0 + 1, z0 + 1] * tmp + c[x0 + 1, y0 + 1, z0 + 1] * xd

    tmp = 1 - yd
    c0 = c00 * tmp + c10 * yd
    c1 = c01 * tmp + c11 * yd

    color = c0 * (1 - zd) + c1 * zd

    if len(color.shape) == 1:
        color = np.expand_dims(color, axis=1)
    return color


def trilinear_coefficients(xyz, origin=np.zeros(3), dx=1.0, dy=1.0, dz=1.0):
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

    tmpX = 1 - xd
    tmpZ = 1 - zd
    tmpY = 1 - yd
    a000 = tmpX * tmpY * tmpZ
    a100 = xd * tmpY * tmpZ
    a010 = tmpX * yd * tmpZ
    a110 = xd * yd * tmpZ
    a001 = tmpX * tmpY * zd
    a101 = xd * tmpY * zd
    a011 = tmpX * yd * zd
    a111 = xd * yd * zd

    avec = np.array([a000, a001, a010, a011, a100, a101, a110, a111]).T
    # avec = np.array([a000, a100, a010, a110, a001, a101, a011, a111])
    return avec.T

def trilinear_interpolation_dot(xyz, c, origin=np.zeros(3), dx=1.0, dy=1.0, dz=1.0):
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

    # a000 = -xd * yd * zd + xd * yd + xd * zd - xd + yd * zd - yd - zd + 1
    # a100 = xd * yd * zd - xd * yd - xd * zd + xd
    # a010 = xd * yd * zd - xd * yd - yd * zd + yd
    # a110 = xd * yd - xd * yd * zd
    # a001 = xd * yd * zd - xd * zd - yd * zd + zd
    # a101 = xd * xd - xd * yd * zd
    # a011 = yd * zd - xd * yd * zd
    # a111 = xd * yd * zd

    tmpX = 1 - xd
    tmpZ = 1 - zd
    tmpY = 1 - yd
    a000 = tmpX * tmpY * tmpZ
    a100 = xd * tmpY * tmpZ
    a010 = tmpX * yd * tmpZ
    a110 = xd * yd * tmpZ
    a001 = tmpX * tmpY * zd
    a101 = xd * tmpY * zd
    a011 = tmpX * yd * zd
    a111 = xd * yd * zd

    # cvec = np.array([c[x0, y0, z0],
    #                  c[x0 + 1, y0, z0],
    #                  c[x0, y0 + 1, z0],
    #                  c[x0 + 1, y0 + 1, z0],
    #                  c[x0, y0, z0 + 1],
    #                  c[x0 + 1, y0, z0 + 1],
    #                  c[x0, y0 + 1, z0 + 1],
    #                  c[x0 + 1, y0 + 1, z0 + 1]])

    x_zeros = np.zeros(x0.shape, dtype=int)
    x_zeros[xd > 0] +=1
    x1 = x0 + x_zeros

    y_zeros = np.zeros(y0.shape, dtype=int)
    y_zeros[yd > 0] +=1
    y1 = y0 + y_zeros

    z_zeros = np.zeros(z0.shape, dtype=int)
    z_zeros[zd > 0] +=1
    z1 = z0 + z_zeros

    # c00 = c[x0, y0, z0] * tmp + c[x1, y0, z0] * xd
    # c01 = c[x0, y1, z0] * tmp + c[x1, y1, z0] * xd
    # c10 = c[x0, y0, z1] * tmp + c[x1, y0, z1] * xd
    # c11 = c[x0, y1, z1] * tmp + c[x1, y1, z1] * xd


    cvec = np.array([c[x0, y0, z0], c[x0, y0, z1],
                     c[x0, y1, z0], c[x0, y1, z1],
                     c[x1, y0, z0], c[x1, y0, z1],
                     c[x1, y1, z0], c[x1, y1, z1]])

    avec = np.array([a000, a001, a010, a011, a100, a101, a110, a111])
    # avec = np.array([a000, a100, a010, a110, a001, a101, a011, a111])

    res = np.einsum('ij,ij...->j...', avec, cvec)

    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=1)
    return res


def trilinear_interpolation_short(vecs, values, origin, delta_voxel):
    # Normalize, transform into origin = (0,0,0) and dx = dy = dz = 1
    # Case when only 1 entry to interpolate, want shape [3, nb]
    xyz = vecs - origin
    xyz = xyz / delta_voxel

    xyz_floor = np.floor(xyz)
    diff = xyz - xyz_floor

    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]
    xyz_floor = xyz_floor.astype(int)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
    tmpX = 1 - xd
    tmpY = 1 - yd
    tmpZ = 1 - zd
    a000 = tmpX * tmpY
    a100 = xd * tmpY
    a010 = tmpX * yd
    a110 = xd * yd

    weights = np.array([a000, a010, a100, a110])
    coeff = np.array([values[x0, y0, z0], values[x0, y0, z0 + 1],
                         values[x0, y0 + 1, z0], values[x0, y0 + 1, z0 + 1],
                         values[x0 + 1, y0, z0], values[x0 + 1, y0, z0 + 1],
                         values[x0 + 1, y0 + 1, z0], values[x0 + 1, y0 + 1, z0 + 1]])

    weights = weights[:, :, None]
    if tmpZ.ndim == 1 and zd.ndim == 1:
        tmpZ = tmpZ[:, None]
        zd = zd[:, None]

    res = np.sum(weights * coeff[[0, 2, 4, 6]], axis=0) * tmpZ + np.sum(weights * coeff[[1, 3, 5, 7]], axis=0) * zd
    return np.sum(weights * coeff[[0, 2, 4, 6]], axis=0) * tmpZ + np.sum(weights * coeff[[1, 3, 5, 7]], axis=0) * zd