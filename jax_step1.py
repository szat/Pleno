import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from trilinear_interpolation import *

#
# nb_samples = 100
# dim = 10
# total_size = 100*100*100
# nb_non_zero = 500000
#
# grid = np.zeros(total_size)
# idx = np.random.choice(total_size, nb_non_zero, replace=False)
# links = np.random.choice(nb_non_zero, nb_non_zero, replace=False)
# grid[idx] = links
# grid = grid.reshape(100, 100, 100).astype(int)
# vecs = np.random.uniform(2, 98, [nb_samples, 3])
# data = np.random.rand(nb_non_zero, dim)
# data[0] = 0
# out_np = trilinear_interpolation_shuffle_zero(vecs, grid, data)
#
# vecs = jnp.array(vecs)
# grid = jnp.array(grid)
# data = jnp.array(data)

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
    coeff = jnp.array([v000, v001, v010, v011, v100, v101, v110, v111])
    weights = jnp.expand_dims(weights, 1)

    out = jnp.sum(weights * coeff[jnp.array([0, 2, 4, 6])], axis=0) * tmpZ + jnp.sum(
        weights * coeff[jnp.array([1, 3, 5, 7])], axis=0) * zd
    return out

# res_jax = []
# for v in vecs:
#     res_jax.append(trilinear_jax(v, grid, data))
#
# res = vmap(trilinear_jax, in_axes=(0, None, None))(vecs, grid, data)
#
#
# res_jax = jnp.concatenate(res_jax)
# res_jax = res_jax.reshape(nb_samples, dim)
# np.testing.assert_allclose(out_np, res)



from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine

path = '/home/adrian/Code/svox2/opt/ckpt/exp2/ckpt.npz'
img_size = 800
# batch_size = 4*1024
# nb_samples = 512
# nb_sh_channels = 3
data = np.load(path, allow_pickle=True)
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

# hack
npy_density_data[0] = 0
npy_sh_data[0] = 0
npy_links[npy_links < 0] = 0
npy_data = np.hstack([npy_density_data, npy_sh_data])

ori = np.load("/home/adrian/Documents/temp/svox_ori.npy")
dir = np.load("/home/adrian/Documents/temp/svox_dir.npy")

# transform to world coords
step_size = 0.5
delta_scale = 1/256
gsz = 256
offset = 0.5 * gsz - 0.5  # 0.5 * 256 - 0.5
scaling = 0.5 * gsz  # 0.5 * 256
ori = offset + ori * scaling

# get the tic samples
spacing = 0.5
box_top = np.ones(3)*256
box_bottom = np.zeros(3)
dir = dir / np.expand_dims(np.linalg.norm(dir, axis=1), 1)
inv = 1/dir
sh = eval_sh_bases_mine(dir)
tmin, tmax = intersect_ray_aabb(ori, inv, box_bottom-0.5, box_top-0.5) # to compute the tmin and tmax, they do a offset of -0.5 first, so add -0.5 to box_min and box_max

mask = tmin < tmax
ori = ori[mask]
dir = dir[mask]
sh = sh[mask]
tmin = tmin[mask]
tmax = tmax[mask]
tmin = tmin.astype(np.float32)
tmax = tmax.astype(np.float32)

colors = np.zeros([800*800, 3])
max_dt = np.max(tmax - tmin)
nb = int(np.ceil(max_dt/spacing))
tmp_tics = np.linspace(0, 1, nb)


iter_range = np.arange(800*800)
list_samples = []
for i in iter_range[:1000]:
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    list_samples.append(samples)


ori = jnp.array(ori)
dir = jnp.array(dir)
tmax = jnp.array(tmax)
tmin = jnp.array(tmin)
tmp_tics = jnp.array(tmp_tics)
def samples_jax(ori, dir, tmin, tmax):
    tics = tmin + tmp_tics * (tmax - tmin)
    samples = ori[None, :] + tics[:, None] * dir[None, :]
    samples = np.clip(samples, 0, 254)
    return samples

res_jax = []
for i in range(1000):
    res_jax.append(samples_jax(ori[i], dir[i], tmin[i], tmax[i], tmp_tics))

res_jax = jnp.concatenate(res_jax)
np.testing.assert_allclose(res_np, res_jax)

res_vmap = vmap(samples_jax, in_axes=(0, 0, 0, 0, None))(ori[:1000], dir[:1000], tmin[:1000], tmax[:1000], tmp_tics)
res_jax = res_jax.reshape(res_vmap.shape)
np.testing.assert_allclose(res_vmap, res_jax)


iter_range = np.arange(800*800)
list_samples = []
list_interp = []
list_colors = []
list_sh = []
for i in iter_range[:2000]:
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    list_samples.append(samples)
    # interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
    ###
    vecs = samples
    values_compressed = npy_data
    origin = np.zeros(3)
    delta_voxel = np.ones(3)
    links = npy_links

    if vecs.ndim == 1:
        vecs = vecs[None, :]

    dims = values_compressed.shape[1]
    nb_samples = vecs.shape[0]
    res = np.zeros([nb_samples, dims])

    xyz = vecs - origin
    xyz = xyz / delta_voxel
    xyz_floor = np.floor(xyz)
    diff = xyz - xyz_floor
    xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]

    xyz_floor = xyz_floor.astype(int)
    x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]

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
    weights = np.array([a000, a010, a100, a110])
    coeff = np.array([v000, v001, v010, v011, v100, v101, v110, v111])
    weights = weights[:, :, None]
    if tmpZ.ndim == 1 and zd.ndim == 1:
        tmpZ = tmpZ[:, None]
        zd = zd[:, None]

    out = np.sum(weights * coeff[[0, 2, 4, 6]], axis=0) * tmpZ + np.sum(weights * coeff[[1, 3, 5, 7]], axis=0) * zd
    interp = out
    ###
    list_interp.append(interp)

    sigma = interp[:, :1]
    rgb = interp[:, 1:]

    sigma = np.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)
    list_sh.append(sh[i])
    sh_ray = sh[i][None, None, :]
    rgb = rgb * sh_ray
    rgb = np.sum(rgb, axis=2)
    rgb = rgb + 0.5 #correction 1
    rgb = np.clip(rgb, a_min=0.0, a_max=100000)
    tmp = step_size * sigma * delta_scale
    # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
    var = 1 - np.exp(-tmp)
    Ti = np.exp(np.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb = coefs * rgb
    rgb = np.sum(rgb, axis=0)
    colors[i] = rgb
    list_colors.append(rgb)
    print(i)


def aggregate_jax(interp, sh):
    sigma = interp[:, :1]
    rgb = interp[:, 1:]

    sigma = jnp.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)

    sh_ray = sh[None, None, :]
    rgb = rgb * sh_ray
    rgb = jnp.sum(rgb, axis=2)
    rgb = rgb + 0.5 #correction 1
    rgb = jnp.clip(rgb, a_min=0.0, a_max=100000)
    tmp = step_size * sigma * delta_scale
    # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
    var = 1 - jnp.exp(-tmp)
    Ti = jnp.exp(np.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb = coefs * rgb
    rgb = jnp.sum(rgb, axis=0)
    return rgb

out_jax = []
for i in range(len(list_interp)):
    interp = jnp.array(list_interp[i])
    sh = jnp.array(list_sh[i])

    rgb = aggregate_jax(interp, sh)
    out_jax.append(rgb)


interp_np = np.concatenate(list_interp)
interp_np = interp_np.reshape(len(list_interp), 727, 28)

sh_np = np.concatenate(list_sh)
sh_np = sh_np.reshape(len(list_sh), 9)

res_vmap = vmap(aggregate_jax, in_axes=(0, 0))(interp_np, sh_np)




def jax_it(ori, dir, tmin, tmax, links, data, sh):
    samples = samples_jax(ori, dir, tmin, tmax)
    interp = vmap(trilinear_jax, in_axes=(0, None, None))(samples, links, data)
    rgb = aggregate_jax(interp, sh)
    return rgb

ori = jnp.array(ori[:2000])
dir = jnp.array(dir[:2000])
tmin = jnp.array(tmin[:2000])
tmax = jnp.array(tmax[:2000])
links = jnp.array(npy_links)
data = jnp.array(npy_data)
sh = jnp.array(sh[:2000])
tmp = vmap(jax_it, in_axes=(0, 0, 0, 0, None, None, 0))(ori, dir, tmin, tmax, links, data, sh)

list_colors = np.concatenate(list_colors)
list_colors = list_colors.reshape(2000, 3)
np.testing.assert_allclose(tmp, list_colors)


