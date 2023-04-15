# import os, sys
# import numpy as np
# # import imageio as im
# import cv2  # resize images with float support
# from scipy import ndimage  # gaussian blur
# import time
# import math
import numpy as np

from scipy.special import sph_harm

K_CONST = np.array([0.28209479, 0.48860251, 0.48860251, 0.48860251, 1.09254843,
       1.09254843, 0.31539157, 1.09254843, 0.54627422])


# My code
def sh_cartesian(xyz):
    # K = np.array([(1 / 2) * np.sqrt(1 / np.pi),
    #               np.sqrt(3 / (4 * np.pi)), np.sqrt(3 / (4 * np.pi)), np.sqrt(3 / (4 * np.pi)),
    #               (1 / 2) * np.sqrt(15 / np.pi), (1 / 2) * np.sqrt(15 / np.pi), (1 / 4) * np.sqrt(5 / np.pi),
    #               (1 / 2) * np.sqrt(15 / np.pi), (1 / 4) * np.sqrt(15 / np.pi)])
    K = K_CONST
    if xyz.ndim == 1:
        r = np.linalg.norm(xyz)
        xyz = xyz / r
        x, y, z = xyz[0], xyz[1], xyz[2]
        vec = np.array([1, y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2])
        return vec * K
    else:
        r = np.linalg.norm(xyz, axis=1)
        r = np.expand_dims(r, 1)
        xyz = xyz / r
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ones = np.ones(x.shape)
        vec = np.vstack([ones, y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2]).T
        return vec * K


# My code
def sh_spherical(theta, phi):
    root3 = np.sqrt(3 / np.pi)
    root5 = np.sqrt(15 / np.pi)
    K = np.array([0.5 * np.sqrt(1 / np.pi),
                  0.5 * root3,
                  0.5 * root3,
                  0.5 * root3,
                  0.5 * root5,
                  0.5 * root5,
                  0.25 * np.sqrt(5 / np.pi),
                  0.5 * root5,
                  0.25 * root5])

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    ones = np.ones(x.shape)
    legendre = np.array([ones, y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y]).T
    res = legendre * K
    if res.shape == (9,):
        res = np.expand_dims(res, 0)
    return res


# From the book "Gritty Details", translated from C++ using ChatGPT
def K(l, m):
    temp = ((2.0 * l + 1.0) * np.math.factorial(l - m)) / (4.0 * np.pi * np.math.factorial(l + m))
    return np.sqrt(temp)


def P(l, m, x):
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def SH(l, m, theta, phi):
    sqrt2 = np.sqrt(2.0)
    cos = np.cos
    sin = np.sin
    if m == 0:
        return K(l, 0) * P(l, m, np.cos(theta))
    elif m > 0:
        return (-1) ** m * sqrt2 * K(l, m) * cos(m * phi) * P(l, m, np.cos(theta))
    else:
        return (-1) ** m * sqrt2 * K(l, -m) * sin(-m * phi) * P(l, -m, np.cos(theta))


def sh_gritty_book(theta, phi):
    return np.array(
        [SH(0, 0, theta, phi), SH(1, -1, theta, phi), SH(1, 0, theta, phi), SH(1, 1, theta, phi), SH(2, -2, theta, phi),
         SH(2, -1, theta, phi), SH(2, 0, theta, phi), SH(2, 1, theta, phi), SH(2, 2, theta, phi)])


# From ChatGPT using scipy, IT SEEMS THETA AND PHI ARE REVERSED HERE
def real_sph_harm_scipy(l, m, theta, phi):
    if m > 0:
        return (-1) ** m * np.sqrt(2) * np.real(sph_harm(m, l, theta, phi))
    elif m < 0:
        return (-1) ** m * np.sqrt(2) * np.imag(sph_harm(-m, l, theta, phi))
    else:
        return np.real(sph_harm(0, l, theta, phi))


# IT SEEMS THETA AND PHI ARE REVERSED HERE
def real_sph_harm_vec_scipy(theta, phi):
    Y00 = real_sph_harm_scipy(0, 0, theta, phi)
    Y1m = real_sph_harm_scipy(1, -1, theta, phi)
    Y10 = real_sph_harm_scipy(1, 0, theta, phi)
    Y1p = real_sph_harm_scipy(1, 1, theta, phi)
    Y2m2 = real_sph_harm_scipy(2, -2, theta, phi)
    Y2m1 = real_sph_harm_scipy(2, -1, theta, phi)
    Y20 = real_sph_harm_scipy(2, 0, theta, phi)
    Y2p1 = real_sph_harm_scipy(2, 1, theta, phi)
    Y2p2 = real_sph_harm_scipy(2, 2, theta, phi)
    return np.array([Y00, Y1m, Y10, Y1p, Y2m2, Y2m1, Y20, Y2p1, Y2p2])


## From the svox paper
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.array([
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
])
SH_C3 = np.array([
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
])
SH_C4 = np.array([
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
])

MAX_SH_BASIS = 10

def eval_sh_bases_mine(dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: np.ndarray (..., 3) unit directions

    :return: np.ndarray (..., basis_dim)
    """
    basis_dim = 9
    result = np.empty([dirs.shape[0], basis_dim], dtype=dirs.dtype)
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


# def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
#     """
#     Evaluate spherical harmonics bases at unit directions,
#     without taking linear combination.
#     At each point, the final result may the be
#     obtained through simple multiplication.
#
#     :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
#     :param dirs: torch.Tensor (..., 3) unit directions
#
#     :return: torch.Tensor (..., basis_dim)
#     """
#     result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
#     result[..., 0] = SH_C0
#     if basis_dim > 1:
#         x, y, z = dirs.unbind(-1)
#         result[..., 1] = -SH_C1 * y;
#         result[..., 2] = SH_C1 * z;
#         result[..., 3] = -SH_C1 * x;
#         if basis_dim > 4:
#             xx, yy, zz = x * x, y * y, z * z
#             xy, yz, xz = x * y, y * z, x * z
#             result[..., 4] = SH_C2[0] * xy;
#             result[..., 5] = SH_C2[1] * yz;
#             result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
#             result[..., 7] = SH_C2[3] * xz;
#             result[..., 8] = SH_C2[4] * (xx - yy);
#
#             if basis_dim > 9:
#                 result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
#                 result[..., 10] = SH_C3[1] * xy * z;
#                 result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
#                 result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
#                 result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
#                 result[..., 14] = SH_C3[5] * z * (xx - yy);
#                 result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);
#
#                 if basis_dim > 16:
#                     result[..., 16] = SH_C4[0] * xy * (xx - yy);
#                     result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
#                     result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
#                     result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
#                     result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
#                     result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
#                     result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
#                     result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
#                     result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
#     return result
