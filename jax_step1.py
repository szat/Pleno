import jax.numpy as jnp
from jax import grad, jit, vmap

from trilinear_interpolation import *

nb_samples = 100
dim = 10
total_size = 100*100*100
nb_non_zero = 500000

grid = np.zeros(total_size)
idx = np.random.choice(total_size, nb_non_zero, replace=False)
links = np.random.choice(nb_non_zero, nb_non_zero, replace=False)
grid[idx] = links
grid = grid.reshape(100, 100, 100).astype(int)
vecs = np.random.uniform(2, 98, [nb_samples, 3])
data = np.random.rand(nb_non_zero, dim)
data[0] = 0
out_np = trilinear_interpolation_shuffle_zero(vecs, grid, data)

vecs = jnp.array(vecs)
grid = jnp.array(grid)
data = jnp.array(data)

def trilinear_jax(vecs, links, values_compressed, origin=jnp.zeros(3), delta_voxel=jnp.ones(3)):
    xyz = vecs - origin
    xyz = xyz / delta_voxel
    xyz_floor = jnp.floor(xyz)
    diff = xyz - xyz_floor
    xd, yd, zd = diff[0], diff[1], diff[2]
    xyz_floor = xyz_floor.astype(int)
    x0, y0, z0 = xyz_floor[0], xyz_floor[1], xyz_floor[2]
    l000 = links[x0, y0, z0]
    l100 = links[x0 + 1, y0, z0]
    l010 = links[x0, y0 + 1, z0]
    l001 = links[x0, y0, z0 + 1]
    l110 = links[x0 + 1, y0 + 1, z0]
    l011 = links[x0, y0 + 1, z0 + 1]
    l101 = links[x0 + 1, y0, z0 + 1]
    l111 = links[x0 + 1, y0 + 1, z0 + 1]
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
    # return weights
    coeff = jnp.array([v000, v001, v010, v011, v100, v101, v110, v111])
    weights = jnp.expand_dims(weights, 1)
    # tmpZ = jnp.expand_dims(tmpZ, 1)
    # zd = jnp.expand_dims(zd, 1)

    out = jnp.sum(weights * coeff[jnp.array([0, 2, 4, 6])], axis=0) * tmpZ + jnp.sum(
        weights * coeff[jnp.array([1, 3, 5, 7])], axis=0) * zd
    return out

res_jax = []
for v in vecs:
    res_jax.append(trilinear_jax(v, grid, data))

res = vmap(trilinear_jax, in_axes=(0, None, None))(vecs, grid, data)


res_jax = jnp.concatenate(res_jax)
res_jax = res_jax.reshape(nb_samples, dim)
np.testing.assert_allclose(out_np, res)

#
# from sampling_branch import intersect_ray_aabb
# from spherical_harmonics import eval_sh_bases_mine
#
# path = '/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz'
# img_size = 800
# # batch_size = 4*1024
# # nb_samples = 512
# # nb_sh_channels = 3
# data = np.load(path, allow_pickle=True)
# npy_radius = data['radius']
# npy_center = data['center']
# npy_links = data['links']
# npy_density_data = data['density_data']
# npy_sh_data = data['sh_data']
# npy_basis_type = data['basis_type']
#
# # hack
# npy_density_data[0] = 0
# npy_sh_data[0] = 0
# npy_links[npy_links < 0] = 0
# npy_data = np.hstack([npy_density_data, npy_sh_data])
#
# ori = np.load("/home/adrian/Documents/temp/svox_ori.npy")
# dir = np.load("/home/adrian/Documents/temp/svox_dir.npy")
#
# # transform to world coords
# step_size = 0.5
# delta_scale = 1/256
# gsz = 256
# offset = 0.5 * gsz - 0.5  # 0.5 * 256 - 0.5
# scaling = 0.5 * gsz  # 0.5 * 256
# ori = offset + ori * scaling
#
# # get the tic samples
# spacing = 0.5
# box_top = np.ones(3)*256
# box_bottom = np.zeros(3)
# dir = dir / np.expand_dims(np.linalg.norm(dir, axis=1), 1)
# inv = 1/dir
# sh = eval_sh_bases_mine(dir)
# tmin, tmax = intersect_ray_aabb(ori, inv, box_bottom-0.5, box_top-0.5) # to compute the tmin and tmax, they do a offset of -0.5 first, so add -0.5 to box_min and box_max
#
# mask = tmin < tmax
# ori = ori[mask]
# dir = dir[mask]
# sh = sh[mask]
# tmin = tmin[mask]
# tmax = tmax[mask]
# tmin = tmin.astype(np.float32)
# tmax = tmax.astype(np.float32)
#
# colors = np.zeros([800*800, 3])
# max_dt = np.max(tmax - tmin)
# nb = int(np.ceil(max_dt/spacing))
# tmp_tics = np.linspace(0, 1, nb)
# res_np = []
