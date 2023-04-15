import numpy as np
from trilinear_interpolation import *

# xdim, ydim, zdim = 10, 10, 10
# dx, dy, dz = 1, 1, 1
# nb_rays = 5
# nb_samples = 7
# nb_cameras = 6
# cameras = np.random.rand(nb_cameras, 6) # position and orientation

# http://psgraphics.blogspot.com/2016/02/new-simple-ray-box-test-from-andrew.html
def intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max):
    # considers the boundary of the volume as NON intersecting, if tmax <= tmin then NO intersection
    if ray_origin.ndim == 1:
        ray_origin = np.expand_dims(ray_origin, 0)
        ray_inv_dir = np.expand_dims(ray_inv_dir, 0)
    tmin = np.ones(len(ray_origin)) * -np.inf
    tmax = np.ones(len(ray_origin)) * np.inf
    t0 = (box_min - ray_origin) * ray_inv_dir
    t1 = (box_max - ray_origin) * ray_inv_dir
    tsmaller = np.nanmin([t0, t1], axis=0)
    tbigger = np.nanmax([t0, t1], axis=0)
    tmin = np.max([tmin, np.max(tsmaller, axis=1)], axis=0)
    tmax = np.min([tmax, np.min(tbigger, axis=1)], axis=0)
    return tmin, tmax


def rays_to_samples(rays, spacing, box_top, box_bottom):
    rays_ori = rays[:, :3]
    rays_dir = rays[:, 3:]
    rays_dir = rays_dir / np.expand_dims(np.linalg.norm(rays_dir, axis=1), 1)
    rays_inv = 1/rays_dir
    tmin, tmax = intersect_ray_aabb(rays_ori, rays_inv, box_bottom, box_top)
    mask = tmin < tmax
    rays = rays[mask]
    tmin = tmin[mask]
    tmax = tmax[mask]
    tics = []
    for i in range(len(tmin)):
        tics.append(np.arange(tmin[i], tmax[i], spacing))
    samples = []
    for i, t in enumerate(tics):
        ori = rays_ori[i]
        dir = rays_dir[i]
        ori = np.expand_dims(ori, axis=1)
        dir = np.expand_dims(dir, axis=1)
        samples.append((ori + t * dir).T)
    return samples, rays


def samples_to_interpolation_coeffs(samples, dx=1.0, dy=1.0, dz=1.0):
    coefs = []
    origin = np.zeros(3)
    for s in samples:
        coefs.append(trilinear_coefficients(s, origin, dx, dy, dz))
    return coefs


# rays = np.random.rand(nb_rays, 3) # rays * xyz (orientation)
# samples = np.random.rand(nb_rays, nb_samples, 3) # rays * samples per ray * xyz (position)
# interp_coefs = np.random.rand(nb_rays, nb_samples, 8) # constant at a point
# spherical_harmonics = np.random.rand(nb_rays, 9) # constant on a ray
# grid = np.random.rand(xdim, ydim, zdim, 9*3) # 3*9=27 sh coefficients at every vertex
# opacity = np.random.rand(xdim, ydim, zdim) # one opacity value at every vertex
#
# # rays -> samples
# # rays -> spherical_harmonics
# # samples -> interp_coefs
# # grid(sample) -> 8 x 27 coefs of the spherical harmonics at the surrounding corners
# # <grid(sample), interp_coefs> -> inner prod, values per harmonic at sample
# # sum up inner prod * opacity along ray to get value
# # we want to optimize on the values in grid, so we want the derivative w.r.t. the values in grid, which is a_iS(r)
# output = np.random.rand(nb_rays, 3) # rgb per ray
