import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from trilinear_interpolation import *

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

#### end setup

# iter_range = np.arange(800*800)
iter_range = np.random.choice(800*800, 10000, replace=False)
res_np = []
for i in iter_range:
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    res_np.append(samples)

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
for i in iter_range:
    res_jax.append(samples_jax(ori[i], dir[i], tmin[i], tmax[i]))

res_vmap = vmap(samples_jax, in_axes=(0, 0, 0, 0))(ori[iter_range], dir[iter_range], tmin[iter_range], tmax[iter_range])

res_jax = jnp.concatenate(res_jax)
res_jax = res_jax.reshape(res_vmap.shape)
np.testing.assert_allclose(res_vmap, res_jax)

res_np = np.concatenate(res_np)
res_np = res_np.reshape(res_vmap.shape)
np.testing.assert_allclose(res_vmap, res_np)
