import numpy as np
from trilinear_interpolation import *
from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

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

colors = np.zeros([800*800, 3])
max_dt = np.max(tmax - tmin)
nb = int(np.ceil(max_dt/spacing))
tmp_tics = np.linspace(0, 1, nb)

iter_range = np.random.choice(800*800, 10000, replace=False)

list_samples = []
list_interp = []
list_colors = []
list_sh = []

for i in iter_range:
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)

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
    list_colors.append(rgb)

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

res_jax = []
for i in range(len(list_interp)):
    interp = jnp.array(list_interp[i])
    sh = jnp.array(list_sh[i])

    rgb = aggregate_jax(interp, sh)
    res_jax.append(rgb)

interp_jnp = jnp.concatenate(list_interp)
interp_jnp = interp_jnp.reshape(len(list_interp), 727, 28)

sh_jnp = jnp.concatenate(list_sh)
sh_jnp = sh_jnp.reshape(len(list_interp), 9)

res_vmap = vmap(aggregate_jax, in_axes=(0, 0))(interp_jnp, sh_jnp)

res_jax = jnp.concatenate(res_jax)
res_jax = res_jax.reshape(res_vmap.shape)
np.testing.assert_allclose(res_vmap, res_jax)

res_np = np.concatenate(list_colors)
res_np = res_np.reshape(res_vmap.shape)
np.testing.assert_allclose(res_vmap, res_np)



