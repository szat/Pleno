from jax import grad, jit, vmap
from jax.config import config
import jax
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

def trilinear_interpolation_to_vmap(vecs, links, values_compressed):
    # xyz = vecs - origin
    # xyz = xyz / delta_voxel
    xyz = vecs
    xyz_floor = jnp.floor(xyz)
    xd, yd, zd = xyz - xyz_floor
    x0, y0, z0 = xyz_floor.astype(int)

    l000 = links[x0, y0, z0]
    l100 = links[x0+1, y0, z0]
    l010 = links[x0, y0+1, z0]
    l001 = links[x0, y0, z0+1]
    l110 = links[x0+1, y0+1, z0]
    l011 = links[x0, y0+1, z0+1]
    l101 = links[x0+1, y0, z0+1]
    l111 = links[x0+1, y0+1, z0+1]

    v000 = values_compressed[l000]
    v100 = values_compressed[l100]
    v010 = values_compressed[l010]
    v001 = values_compressed[l001]
    v110 = values_compressed[l110]
    v011 = values_compressed[l011]
    v101 = values_compressed[l101]
    v111 = values_compressed[l111]

    tmpX = 1 - xd
    tmpY = 1 - yd
    tmpZ = 1 - zd
    a000 = tmpX * tmpY
    a100 = xd * tmpY
    a010 = tmpX * yd
    a110 = xd * yd
    weights = jnp.array([a000, a010, a100, a110])

    coeff = jnp.array([v000, v001, v010, v011, v100, v101, v110, v111])
    # return coeff, weights, tmpZ, zd
    weights = weights[:, None]

    out = jnp.sum(weights * coeff[[0, 2, 4, 6], :], axis=0) * tmpZ + jnp.sum(weights * coeff[[1, 3, 5, 7], :], axis=0) * zd
    out = out[None, :]
    return out

jit_trilinear_interp = jit(vmap(trilinear_interpolation_to_vmap, in_axes=(0, None, None)))

def rotation_align(from_vec, to_vec):
    assert from_vec.shape == to_vec.shape, "from_vec and to_vec need to be of the same shape"
    if from_vec.ndim == 1:
        v = np.cross(from_vec, to_vec)
        # c = np.einsum('ij,ij...->i...', from_vec, to_vec)
        c = np.dot(from_vec, to_vec)
        if np.all(v == np.zeros(3)) and c > 0:
            return np.eye(3)
        if np.all(v == np.zeros(3)) and c < 0:
            return -np.eye(3)
        k = 1.0 / (1.0 + c)
        return np.array([[v[0]**2 * k + c,    v[0]*v[1]*k - v[2], v[0]*v[2]*k + v[1]],
                         [v[0]*v[1]*k + v[2], v[1]**2 * k + c,    v[1]*v[2]*k - v[0]],
                         [v[0]*v[2]*k - v[1], v[1]*v[2]*k + v[0], v[2]**2 * k + c   ]])
    if from_vec.ndim == 2:
        v = np.cross(from_vec, to_vec)
        c = np.einsum('ij,ij...->i...', from_vec, to_vec)
        k = 1.0 / (1.0 + c)
        out = np.array([[v[:, 0]**2 * k + c,    v[:, 0]*v[:, 1]*k - v[:, 2], v[:, 0]*v[:, 2]*k + v[:, 1]],
                         [v[:, 0]*v[:, 1]*k + v[:, 2], v[:, 1]**2 * k + c,    v[:, 1]*v[:, 2]*k - v[:, 0]],
                         [v[:, 0]*v[:, 2]*k - v[:, 1], v[:, 1]*v[:, 2]*k + v[:, 0], v[:, 2]**2 * k + c   ]])
        out = np.einsum('ijk->kij', out) # rearrange dimensions
        bool_flag_identity = np.all(v == np.zeros(3), axis=1) * c > 0
        bool_flag_reverse = np.all(v == np.zeros(3), axis=1) * c < 0
        out[bool_flag_identity] = np.eye(3)
        out[bool_flag_reverse] = -np.eye(3)
        return out


class Camera:
    def __init__(self,
                 origin=np.zeros(3),
                 orientation=np.array([0, 0, 1]),
                 dist_plane=1,
                 length_x=0.640,
                 length_y=0.480,
                 pixels_x=640,
                 pixels_y=480):
        self.origin = origin
        self.orientation = orientation
        self.dist_plane = dist_plane
        self.length_x = length_x
        self.length_y = length_y
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y


# def get_arrow(start_point, end_point, color=[0.3, 0.3, 0.3], thickness=1):
#     vec = end_point - start_point
#     norm = np.linalg.norm(vec)
#     cone_height = norm * 0.2
#     cylinder_height = norm * 0.8
#     cone_radius = 0.2 * thickness
#     cylinder_radius = 0.1 * thickness
#     arrow = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
#                                                      cone_height=cone_height,
#                                                      cylinder_radius=cylinder_radius,
#                                                      cylinder_height=cylinder_height)
#     vec = vec / norm
#     R = rotation_align(np.array([0, 0, 1]), vec)
#     arrow.rotate(R, center=np.zeros(3))
#     arrow.translate(start_point)
#     arrow.compute_vertex_normals()
#     arrow.paint_uniform_color(color)
#     return arrow


def get_camera_vectors(camera: Camera):
    z = np.array([0, 0, 1])
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    # Put into world coordinates
    R = rotation_align(z, camera.orientation)
    z = np.dot(R, z)
    x = np.dot(R, x)
    y = np.dot(R, y)

    # Camera template
    z = z * camera.dist_plane
    x = x * camera.length_x
    y = y * camera.length_y

    # z = z + camera.origin
    # x = x + camera.origin
    # y = y + camera.origin
    return z, x, y

def get_camera_rays(camera: Camera):
    z, x, y = get_camera_vectors(camera)
    tics_x = np.expand_dims(np.linspace(-1, 1, camera.pixels_x), 1)
    tics_y = np.expand_dims(np.linspace(-1, 1, camera.pixels_y), 1)

    xx = tics_x * x
    yy = tics_y * y
    xx = np.expand_dims(xx, 0)
    yy = np.expand_dims(yy, 1)
    rays = xx + yy
    zz = np.expand_dims(z, [0, 1])
    rays = rays + zz
    rays = rays / np.expand_dims(np.linalg.norm(rays, axis=2), 2)
    return rays

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