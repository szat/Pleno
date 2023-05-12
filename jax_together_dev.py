import numpy as np
from trilinear_interpolation import *
from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

def samples_jax(ori, dir, tmin, tmax, tmp_tics):
    tics = tmin + tmp_tics * (tmax - tmin)
    samples = ori[None, :] + tics[:, None] * dir[None, :]
    samples = np.clip(samples, 0, 254)
    return samples

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

iter_range = np.random.choice(800*800, 2000, replace=False)

# list_samples = []
# list_interp = []
list_colors = []
# list_sh = []

for i in iter_range:
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)

    # list_interp.append(interp)

    sigma = interp[:, :1]
    rgb = interp[:, 1:]

    sigma = np.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)

    # list_sh.append(sh[i])

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


def jax_it(ori, dir, tmin, tmax, links, data, sh, tics):
    samples = samples_jax(ori, dir, tmin, tmax, tics)
    interp = vmap(trilinear_jax, in_axes=(0, None, None))(samples, links, data)
    rgb = aggregate_jax(interp, sh)
    return rgb

ori_jnp = jnp.array(ori[iter_range])
dir_jnp = jnp.array(dir[iter_range])
tmin_jnp = jnp.array(tmin[iter_range])
tmax_jnp = jnp.array(tmax[iter_range])
sh_jnp = jnp.array(sh[iter_range])
links_jnp = jnp.array(npy_links)
data_jnp = jnp.array(npy_data)
tics_jnp = jnp.array(tmp_tics)

res_vmap = vmap(jax_it, in_axes=(0, 0, 0, 0, None, None, 0, None))(ori_jnp, dir_jnp, tmin_jnp, tmax_jnp, links_jnp, data_jnp, sh_jnp, tics_jnp)
res_np = np.concatenate(list_colors)
res_np = res_np.reshape(res_vmap.shape)
np.testing.assert_allclose(res_vmap, res_np)

### try the full image
colors = np.zeros([800*800, 3])
for i in range(800*800):
    tics = tmin[i] + tmp_tics * (tmax[i] - tmin[i])
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
    sigma = interp[:, :1]
    rgb = interp[:, 1:]
    sigma = np.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)
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
    print(i)


batch_size = 2000
total_size = 800*800
res = []
i = 1
links_jnp = jnp.array(npy_links)
data_jnp = jnp.array(npy_data)
tics_jnp = jnp.array(tmp_tics)
while i * batch_size < total_size:
    f = (i-1)*batch_size
    t = i*batch_size
    ori_jnp = jnp.array(ori[f:t])
    dir_jnp = jnp.array(dir[f:t])
    tmin_jnp = jnp.array(tmin[f:t])
    tmax_jnp = jnp.array(tmax[f:t])
    sh_jnp = jnp.array(sh[f:t])
    out = vmap(jax_it, in_axes=(0, 0, 0, 0, None, None, 0, None))(ori_jnp, dir_jnp, tmin_jnp, tmax_jnp, links_jnp, data_jnp, sh_jnp, tics_jnp)
    res.append(out)
    i += 1
    print(i)
f = (i - 1) * batch_size

ori_jnp = jnp.array(ori[f:])
dir_jnp = jnp.array(dir[f:])
tmin_jnp = jnp.array(tmin[f:])
tmax_jnp = jnp.array(tmax[f:])
out = vmap(jax_it, in_axes=(0, 0, 0, 0, None, None, 0, None))(ori_jnp, dir_jnp, tmin_jnp, tmax_jnp, links_jnp, data_jnp, sh_jnp, tics_jnp)
res.append(out)

res_jax = np.concatenate(res)
np.testing.assert_allclose(res_jax, colors)

img = res_jax.reshape([800,800,3])
# img = colors.reshape([800,800,3])
import cv2
# if nb_sh_channels == 2:
#     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
img = (img * 255).astype(np.uint8)
# if nb_sh_channels == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
