from typing import Tuple
import numpy as np

from spherical_harmonics import sh_cartesian, sh_spherical


# relative 8 ijk-neighbours:
delta_ijk = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                      (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]).reshape((8, 3))


def rays_to_frustrum(ray_origins: np.ndarray, ray_dir_vecs: np.ndarray, samples: np.ndarray,
                     delta_voxel: np.array = np.array([1, 1, 1])) -> np.ndarray:
    """Computes the bottom-left-closest voxel (integer coordinates) of for the given rays and samples.

    A ray is defined by its origin (x0, y0, z0) and its direction vector (xd, yd, zd)

    A sample along a ray is defined by its distance s to the ray origin:
           (x0, y0, z0) + s * (xd, yd, zd)

    Args: 
        ray_origins (np.ndarray): of shape (nb_rays, 3), second dim gives xyz
        ray_dir_vecs (np.ndarray): of shape (nb_rays, 3), second dim gives (xd,yd,zd)
        samples (np.ndarray): of shape (nb_rays, nb_samples), second dim gives the distance samples
                 for the ray as defined above
        delta_voxel (np.array): of length 3, defining (dx, dy, dz) for voxels. Default (1, 1, 1)

    Returns: np.ndarray of shape (nb_rays, nb_samples, 8, 3), fourth dim gives the integer 3d-coordinates ijk
                    of 8 neightbours of bottom-left-closest voxel containing resp. sample
    """

    n_rays = ray_origins.shape[0]
    assert n_rays == ray_dir_vecs.shape[0]
    assert n_rays == samples.shape[0]

    # normalize ray_dir_vecs
    r = np.linalg.norm(ray_dir_vecs, axis=1)
    r = np.expand_dims(r, 1)
    assert np.all(r)
    ray_dir_vecs = ray_dir_vecs / r

    samples = np.expand_dims(samples, axis=2)
    ray_origins = np.expand_dims(ray_origins, axis=1)
    ray_dir_vecs = np.expand_dims(ray_dir_vecs, axis=1)
    sample_points = ray_origins + samples * ray_dir_vecs

    # get ijk coordinates according to (dx, dy, dz):
    frustum = np.floor(sample_points / delta_voxel).astype(int)

    # compute integer index coords of 8 neighbours:
    frustum = np.expand_dims(frustum, axis=2)
    frustum = frustum + delta_ijk

    ray_origins = np.expand_dims(ray_origins, axis=2)
    dir_vec_neighs = frustum * delta_voxel - ray_origins 

    return frustum, dir_vec_neighs


def frustrum_to_harmonics_fixed_dir(frustum: np.ndarray, ray_dirs: np.ndarray,
                                    grid: np.ndarray, opacity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """computes the multiplication of harmonics (evaluated at a fixed direction per ray and the model 
    coefficients of the 8 integer neighbours defined by frustum per sample.
    
    Args: 
        frustum (np.ndarray): of shape (nb_rays, nb_samples, 8, 3), given by rays_to_frustrum function
        ray_dirs (np.ndarray): of shape (nb_rays, 2), second dim gives (theta, phi)
        grid (np.ndarray): of shape (xdim, ydim zdim, 9) giving model 9 harmonic coeficients per voxel
        opacity (np.ndarray): of shape (xdim, ydim, zdim) giving model opacity
        
    Returns: Tuple[np.ndarray, np.ndarray] resp. neigh_harmonics and neigh_densities. First array is of shape
    (nb_rays, nb_samples, 8, 9) and gives the evaluated 9 harmonics weigthed by model coefficients for each
    RGB-channel and 8 neighbours of each sample of each ray. Second array is of shape (nb_rays, nb_samples, 8) and
    gives the model opacities at all 8 neighbours of each sample of each ray.
    """
    
    assert frustum.shape[0] == ray_dirs.shape[0]
    
    # retrieve model coefficients of harmonics at 8 neigbour indexes:
    neigh_harmonics_coeff = grid[tuple(frustum.T)]
    # resulting array of shape nb_rays x nb_samples x 8 x 9:
    neigh_harmonics_coeff = np.moveaxis(neigh_harmonics_coeff, (0, 1, 2), (2, 1, 0))

    # retrieve model density at 8 neigbour indexes:
    neigh_densities = opacity[tuple(frustum.T)]
    neigh_densities = np.moveaxis(neigh_densities, (0, 1, 2), (2, 1, 0)) # nb_rays x nb_samples x 8

    # evaluations of 8 neighbours harmonics are done in the same ray direction (parallel to it):
    sh = sh_spherical(ray_dirs[:,0], ray_dirs[:,1]) # nb_rays x 9
    sh = np.expand_dims(sh, axis=(2, 3)) # nb_rays x 1 x 1 x 9 
    #sh = np.moveaxis(sh, 1, 4) # nb_rays x 1 x 1 x 9
    
    # weigh harmonics with model grid coefficients at all 8 neighbours
    neigh_harmonics = neigh_harmonics_coeff * sh 

    return neigh_harmonics, neigh_densities


def frustrum_to_harmonics(frustum: np.ndarray, dir_vec_neighs: np.ndarray,
                          grid: np.ndarray, opacity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """computes the multiplication of harmonics (evaluated at a fixed direction per ray and the model 
    coefficients of the 8 integer neighbours defined by frustum per sample.
    
    Args: 
        frustum (np.ndarray): of shape (nb_rays, nb_samples, 8, 3), given by rays_to_frustrum function
        grid (np.ndarray): of shape (xdim, ydim zdim, 9) giving model 9 harmonic coeficients
        opacity (np.ndarray): of shape (xdim, ydim, zdim) giving model opacity
        
    Returns: Tuple[np.ndarray, np.ndarray] resp. neigh_harmonics and neigh_densities. First array is of shape
    (nb_rays, nb_samples, 8, 9) and gives the evaluated 9 harmonics weigthed by model coefficients
    per 8 neighbours of each sample of each ray. Second array is of shape (nb_rays, nb_samples, 8) and
    gives the model opacities at all 8 neighbours of each sample of each ray.
    """
    
    assert frustum.shape == dir_vec_neighs.shape
    
    # retrieve model coefficients of harmonics at 8 neigbour indexes:
    neigh_harmonics_coeff = grid[tuple(frustum.T)]
    # resulting array of shape nb_rays x nb_samples x 8 x 9:
    neigh_harmonics_coeff = np.moveaxis(neigh_harmonics_coeff, (0, 1, 2), (2, 1, 0))

    # retrieve model density at 8 neigbour indexes:
    neigh_densities = opacity[tuple(frustum.T)]
    neigh_densities = np.moveaxis(neigh_densities, (0, 1, 2), (2, 1, 0)) # nb_rays x nb_samples x 8

    # evaluations of 8 neighbours harmonics are done in their own direction:
    sh = sh_cartesian(dir_vec_neighs.reshape((-1, 3))) # (nb_rays * nb_samples * 8) x 9
    sh = sh.reshape((dir_vec_neighs.shape[0], dir_vec_neighs.shape[1], 8, 9)) # nb_rays x nb_samples x 8 x 9
    
    # weigh harmonics with model grid coefficients at all 8 neighbours
    neigh_harmonics = neigh_harmonics_coeff * sh 

    return neigh_harmonics, neigh_densities
