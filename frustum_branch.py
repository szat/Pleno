import numpy as np


xdim, ydim, zdim = 10, 10, 10
dx, dy, dz = 1, 1, 1
nb_rays = 5
nb_samples = 7
nb_cameras = 6
cameras = np.random.rand(nb_cameras, 6) # position and orientation

def rays_to_frustrum(rays, nb_samples):
    return 0


def frustrum_to_harmonics(samples):
    return 0


