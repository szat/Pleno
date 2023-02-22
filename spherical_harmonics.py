import os, sys
import numpy as np
import imageio as im
import cv2  # resize images with float support
from scipy import ndimage  # gaussian blur
import time
import math
import numpy as np

from scipy.special import sph_harm


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


# My code
def sh_cartesian(xyz):
    r = np.linalg.norm(xyz)
    K = np.array([(1 / 2) * np.sqrt(1 / np.pi),
                  np.sqrt(3 / (4 * np.pi)), np.sqrt(3 / (4 * np.pi)), np.sqrt(3 / (4 * np.pi)),
                  (1 / 2) * np.sqrt(15 / np.pi), (1 / 2) * np.sqrt(15 / np.pi), (1 / 4) * np.sqrt(5 / np.pi),
                  (1 / 2) * np.sqrt(15 / np.pi), (1 / 4) * np.sqrt(15 / np.pi)])
    xyz = xyz / r
    x, y, z = xyz[0], xyz[1], xyz[2]

    vec = np.array([1, y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2])
    return vec * K
