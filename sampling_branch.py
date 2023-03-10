import numpy as np


xdim, ydim, zdim = 10, 10, 10
dx, dy, dz = 1, 1, 1
nb_rays = 5
nb_samples = 7
nb_cameras = 6
cameras = np.random.rand(nb_cameras, 6) # position and orientation

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

def rays_to_samples(rays, nb_samples):
    box_top = np.array([(xdim-1)*dx, (ydim-1)*dy, (zdim-1)*dz])
    box_bottom = np.zeros(3)
    samples = np.empty([len(rays), nb_samples, 3])
    rays = np.random.rand(10, 6)
    rays_ori = rays[:, :3]
    rays_dir = rays[:, 3:]
    rays_inv = 1/rays_dir
    tmin, tmax = intersect_ray_aabb(rays_ori, rays_inv, box_bottom, box_top)


    # do the T approach




    return 0

#
# def samples_to_icoeffs(samples):
#     return 0
#
#
# def samples_to_nbh(samples):
#     return 0
#
#
#
# def cameras_to_grid(cameras, xdim, ydim, zdim, xd, yd, zd):
#     # Assume grid is at (0,0,0) with idim being the number of tics, and id the spacing between them.
#     # So if xd = yd = zd = 2.0 and idim = 3, then the grid goes like [0.0, 2.0, 4.0].
#     # Camera is a nb_camera x 6 matrix, first 3 are the positions, last 3 the orientations
#     nb_cam = len(cameras)
#     cam_pos = cameras[:, :3]
#     # Unroll grid into [nb_pts, 3]
#
#     x_ = np.arange(0, xdim*xd, xd, dtype=float)
#     y_ = np.arange(0, ydim*yd, yd, dtype=float)
#     z_ = np.arange(0, zdim*zd, zd, dtype=float)
#
#     x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
#     cam_x = cam_pos[:, 0]
#     cam_y = cam_pos[:, 1]
#     cam_z = cam_pos[:, 2]
#     cam_x = cam_x[:, None, None, None]
#     cam_y = cam_y[:, None, None, None]
#     cam_z = cam_z[:, None, None, None]
#     res = np.empty([nb_cam, 3, xdim, ydim, zdim])
#     res[:, :, :, :, 0] = cam_x - x
#     res[:, :, :, :, 1] = cam_y - y
#     res[:, :, :, :, 2] = cam_z - z
#     return res
#
#
# def ray_to_sampling(rays):
#     nb_rays = len(rays)
#     # Sample along one axis
#     ep = 0.00001
#     a = 0.0 + ep
#     b = (xdim-1)*dx - ep
#     nb_samples = 10
#     x_sampling = np.linspace(a, b, nb_samples)
#     # Later have to check w.r.t. to the orientation
#     # x = B + At
#     B = rays[:,:3] #position
#     A = rays[:,3:] #orientation
#     t = x_sampling[:,None] - B[:,0] #for X
#     t = t / A[:,0]
#     t = t
#     # y = b + at
#     x_sampling = x_sampling[:,None] * np.ones(nb_rays)
#     y_sampling = B[:,1] + A[:,1]*t
#     z_sampling = B[:,2] + A[:,2]*t
#     res = np.empty([nb_rays, nb_samples, 3])
#     res[:,:,0] = x_sampling.T
#     res[:,:,1] = y_sampling.T
#     res[:,:,2] = z_sampling.T
#
#     return res
#
#
# #
# cameras_to_grid = np.random.rand(nb_cameras, xdim, ydim, xdim, 3)
#
#
#
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
