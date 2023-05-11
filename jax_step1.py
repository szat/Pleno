import jax.numpy as jnp
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
res_np = []
# weights_list = []
# coeff_list = []
# l000_list = []
iter = 800*100
for i in range(iter):
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])
    res_np.append(tics)
    # samples = ori[i, None] + tics[:, None] * dir[i, None]
    # samples = np.clip(samples, 0, 254)
    # res_np.append(samples)
res_np = np.concatenate(res_np)

tmin = tmin[:, None]
tmax = tmax[:, None]
ori = jnp.array(ori)
dir = jnp.array(dir)
tmin = jnp.array(tmin)
tmax = jnp.array(tmax)
# tics = jnp.linspace(0, 1, nb)
# tics = tics[:, None]
# origin = jnp.zeros(3)
# delta_voxel = jnp.ones(3)
# sh = jnp.array(sh)
# links = jnp.array(npy_links).astype(int)
# values_compressed = jnp.array(npy_data).astype(jnp.float16)
tmp_tics2 = jnp.array(tmp_tics) #linspace(0, 1, nb)

def step1(ori, dir, tmin, tmax):
    tics = tmin + tmp_tics2 * (tmax - tmin)
    return tics
    # ori = ori[None, :]
    # dir = dir[None, :]
    # tics = tics[:, None]
    # samples = ori + dir * tics
    # return samples
    # samples = jnp.clip(samples, 0, 254)
    # return samples

# res = vmap(step1, in_axes=(0, 0, 0, 0, 0))(ori[:1000], dir[:1000], tmin[:1000], tmax[:1000], sh[:1000])
res = vmap(step1, in_axes=(0, 0, 0, 0))(ori[:iter], dir[:iter], tmin[:iter], tmax[:iter])
res_np = res_np.reshape(80000, nb)
np.testing.assert_allclose(res, res_np)

