import copy
import os
import sys
sys.path.append('.')
import cv2
import numpy as np
from sampling_branch import intersect_ray_aabb
import model

if_torch = False
if if_torch:
    import torch
    torch.cuda.empty_cache()

model_name = "lego"
# path_to_weigths = f"/home/diego/data/nerf/ckpt_syn/256_to_512_fasttv/{model_name}/ckpt.npz"
path_to_weigths = f"/home/adrian/Documents/Nerf/256_to_512_fasttv/{model_name}/ckpt.npz"
img_size = 800
batch_size = 1024*4
nb_samples = 512
nb_sh_channels = 3
size_model = 256
device = "cuda"

data = np.load(path_to_weigths, allow_pickle=True)

# Access data arrays using keys
npy_radius = data['radius']
npy_center = data['center']
npy_links = data['links']
npy_links = npy_links[::2, ::2, ::2] # reduce resolution to half - use only if size_model = 256
npy_density_data = data['density_data']
npy_sh_data = data['sh_data']
npy_basis_type = data['basis_type']

npy_data = np.hstack([npy_density_data, npy_sh_data])

mask = npy_links >= 0
npy_links_m = npy_links[mask]

if if_torch:
    density_matrix = np.zeros((size_model, size_model, size_model, 1), dtype=np.float32)
    density_matrix[mask] = npy_density_data[npy_links_m]
    density_matrix = np.reshape(density_matrix, (size_model, size_model, size_model))
    density_matrix = torch.from_numpy(density_matrix)

    sh_matrix = np.zeros((size_model, size_model, size_model, 27), dtype=np.float16)
    sh_matrix[mask] = npy_sh_data[npy_links_m]
    sh_matrix = np.reshape(sh_matrix, (size_model, size_model, size_model, 27))
    sh_matrix = torch.from_numpy(sh_matrix)

    rf = model.RadianceField(idim=size_model, grid=sh_matrix, opacity=density_matrix,
                             nb_sh_channels=nb_sh_channels, nb_samples=nb_samples,
                             delta_voxel=torch.tensor([1, 1, 1], dtype=torch.float),
                             device=device)

folder = "/home/adrian/Code/Pleno/data_frames/"

rays_origins = np.load(folder + "all_ori.npy")
rays_dirs = np.load(folder + "all_dir.npy")
ray_inv_dirs = np.load(folder + "all_inv.npy")
tmin = np.load(folder + "all_tmin.npy")
tmax = np.load(folder + "all_tmax.npy")
from spherical_harmonics import eval_sh_bases_mine
sh_mine = eval_sh_bases_mine(rays_dirs)

canvas = np.zeros([800, 800])
canvas[::10, ::10] = 1
mask = np.zeros([800, 800])
mask[200:-200, 200:-200] = 1
canvas = canvas * mask
canvas = canvas.flatten()
mask = canvas == 1

valid_rays_origins = rays_origins[mask]
valid_rays_dirs = rays_dirs[mask]
valid_tmin = tmin[mask]
valid_tmax = tmax[mask]
valid_sh = sh_mine[mask]

mask = valid_tmin < valid_tmax
valid_rays_origins = valid_rays_origins[mask]
valid_rays_dirs = valid_rays_dirs[mask]
valid_tmin = valid_tmin[mask]
valid_tmax = valid_tmax[mask]
valid_sh = valid_sh[mask]

if if_torch:
    valid_rays_origins = torch.from_numpy(valid_rays_origins)
    valid_rays_dirs = torch.from_numpy(valid_rays_dirs)
    valid_tmin = torch.from_numpy(valid_tmin)
    valid_tmax = torch.from_numpy(valid_tmax)
    valid_sh = torch.from_numpy(valid_sh)
    get_samples = True
    rendered_rays, extra = rf.render_rays(valid_rays_origins, valid_rays_dirs, valid_tmin, valid_tmax, batch_size, get_samples)
    rendered_rays = rendered_rays.numpy()
    for i, slice in enumerate(extra):
        np.save(folder + "try_" + str(i) + "_" + slice["name"], np.array(slice["value"].cpu()))

    torch_rays = np.array(rendered_rays)
    # torch_samples = np.array(extra[0].cpu())
    # torch_sh = np.array(extra[1].cpu())
    # torch_interp_sh = np.array(extra[2].cpu())
    # torch_interp_opacities = np.array(extra[3].cpu())
else:
    extra = {}
    try_files = [file for file in os.listdir(folder) if file.startswith('try')]
    for file in try_files:
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            extra[filename] = np.load(file_path)

torch_rays = np.load(folder+"torch_rays.npy")

from jax import grad, jit, vmap
from jax.config import config
import jax
import jax.numpy as jnp

config.update("jax_enable_x64", True)

colors = np.zeros([800*800, 3])
max_dt = np.max(tmax - tmin)
nb = 100
step_size = 0.5
delta_scale = 1/256

my_sh = eval_sh_bases_mine(valid_rays_dirs)
np.testing.assert_almost_equal(my_sh, extra["try_2_sh.npy"])

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
jit_interp = jit(vmap(trilinear_interpolation_to_vmap, in_axes=(0, None, None)))

# samples2 = torch_samples.reshape(-1, 3)
sample_points = extra["try_1_sample_points.npy"].reshape(-1, 3)
# sample_points = extra["try_1_sample_points.npy"]
interp = jit_interp(sample_points, npy_links, npy_data)
interp = np.squeeze(interp)
interp_sh_coeffs = interp[:, 1:][None, :, :]
interp_opacities = interp[:, :1][None, :, :]

np.testing.assert_almost_equal(extra["try_3_interp_sh_coeffs.npy"], interp_sh_coeffs)
np.testing.assert_almost_equal(extra["try_4_interp_opacities.npy"], interp_opacities)

interp_opacities = jnp.clip(interp_opacities, a_min=0.0, a_max=100000)

samples = extra["try_0_samples.npy"]
deltas = samples[:, 1:] - samples[:, :-1]

np.testing.assert_almost_equal(extra["try_5_deltas.npy"], deltas)

interp_sh_coeffs = interp_sh_coeffs.reshape((samples.shape[0], samples.shape[1], 3, 9))
interp_opacities = interp_opacities.reshape((samples.shape[0], samples.shape[1], 1))
interp_harmonics = interp_sh_coeffs * my_sh[:, None, None, :]

deltas_times_sigmas = - deltas[:, :, None] * interp_opacities[:, :-1]
np.testing.assert_almost_equal(extra["try_6_deltas_times_sigmas.npy"][:, :, None], deltas_times_sigmas)

cum_weighted_deltas = jnp.cumsum(deltas_times_sigmas, axis=1)
cum_weighted_deltas = jnp.squeeze(cum_weighted_deltas)
tmp = jnp.zeros([1600, 1])
cum_weighted_deltas = jnp.concatenate([tmp, cum_weighted_deltas[:, :-1]], axis=1)
np.testing.assert_almost_equal(extra["try_7_cum_weighted_deltas.npy"], cum_weighted_deltas)

samples_colors = jnp.clip(jnp.sum(interp_harmonics, axis=3) + 0.5, a_min=0.0, a_max=100000)
np.testing.assert_almost_equal(extra["try_8_samples_color.npy"], samples_colors)

deltas_times_sigmas = jnp.squeeze(deltas_times_sigmas)
tmp1 = jnp.exp(cum_weighted_deltas)
tmp2 = 1 - jnp.exp(deltas_times_sigmas)
rays_color = jnp.sum(tmp1[:, :, None] * tmp2[:, :, None] * samples_colors[:, :-1, :], axis=1)

np.testing.assert_almost_equal(extra["try_9_rays_color.npy"], rays_color)

def conv_to_vmap(samples_in, sample_points_in, my_sh_in, npy_links_in, npy_data_in):
    interp = jit_interp(sample_points_in, npy_links_in, npy_data_in)
    # out = interp
    interp = np.squeeze(interp)
    interp_sh_coeffs = interp[:, 1:][None, :, :]
    interp_opacities = interp[:, :1][None, :, :]
    interp_opacities = jnp.clip(interp_opacities, a_min=0.0, a_max=100000)
    deltas = samples_in[1:] - samples_in[:-1]

    interp_sh_coeffs = interp_sh_coeffs.reshape(1, samples.shape[1], 3, 9)
    interp_opacities = interp_opacities.reshape(1, samples.shape[1], 1)
    interp_harmonics = interp_sh_coeffs * my_sh_in[None, None, None, :]

    interp_opacities = jnp.squeeze(interp_opacities)

    deltas_times_sigmas = - deltas * interp_opacities[:-1]

    cum_weighted_deltas = jnp.cumsum(deltas_times_sigmas)
    cum_weighted_deltas = jnp.concatenate([jnp.zeros(1), cum_weighted_deltas[:-1]])

    samples_colors = jnp.clip(jnp.sum(interp_harmonics, axis=3) + 0.5, a_min=0.0, a_max=100000)
    samples_colors = jnp.squeeze(samples_colors)
    deltas_times_sigmas = jnp.squeeze(deltas_times_sigmas)
    tmp1 = jnp.exp(cum_weighted_deltas)
    tmp2 = 1 - jnp.exp(deltas_times_sigmas)
    rays_color = jnp.sum(tmp1[:, None] * tmp2[:, None] * samples_colors[:-1], axis=0)
    out = rays_color
    return out

jit_again = jit(vmap(conv_to_vmap, in_axes=(0, 0, 0, None, None)))

sample_points = sample_points.reshape([1600, 461, 3])
res = jit_again(samples, sample_points, my_sh, npy_links, npy_data)
np.testing.assert_almost_equal(extra["try_9_rays_color.npy"], res)







def main_to_vmap(ori, dir, tmin, sh, max_dt, npy_links, npy_data):
    tics = jnp.linspace(tmin, max_dt + tmin, num=nb, dtype=jnp.float64)
    samples = ori[None, :] + tics[:, None] * dir[None, :]
    samples = jnp.clip(samples, 0, 254)
    interp = jit_interp(samples, npy_links, npy_data)
    interp = jnp.squeeze(interp)

    sigma = interp[:, :1]
    rgb = interp[:, 1:]

    sigma = jnp.clip(sigma, a_min=0.0, a_max=100000)
    rgb = rgb.reshape(-1, 3, 9)
    rgb = rgb * sh[None, None, :]
    rgb = jnp.sum(rgb, axis=2)
    rgb = rgb + 0.5 #correction 1
    rgb = jnp.clip(rgb, a_min=0.0, a_max=100000)
    tmp = step_size * sigma * delta_scale

    var = 1 - jnp.exp(-tmp)
    Ti = jnp.exp(jnp.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb = coefs * rgb
    rgb = jnp.sum(rgb, axis=0)
    return rgb
jit_main = jit(vmap(main_to_vmap, in_axes=(0, 0, 0, 0, None, None, None)))

def main_to_vmap_samples_provided(samples, sh, npy_links, npy_data):
    interp = jit_interp(samples, npy_links, npy_data)
    interp = jnp.squeeze(interp)

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

    var = 1 - jnp.exp(-tmp)
    Ti = jnp.exp(jnp.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb = coefs * rgb
    rgb = jnp.sum(rgb, axis=0)
    return rgb
jit_samples_main = jit(vmap(main_to_vmap_samples_provided, in_axes=(0, 0, None, None)))


# ori = jax.device_put(jnp.array(valid_rays_origins))
# dir = jax.device_put(jnp.array(valid_rays_dirs))
# tmin = jax.device_put(jnp.array(valid_tmin))
npy_links = jax.device_put(jnp.array(npy_links))
npy_data = jax.device_put(jnp.array(npy_data))
torch_samples = jax.device_put(jnp.array(torch_samples))
valid_sh = jax.device_put(jnp.array(valid_sh))
# sh_mine = eval_sh_bases_mine(np.array(valid_rays_dirs))
# sh_mine = jax.device_put(jnp.array(sh_mine))

res = jit_samples_main(torch_samples, valid_sh, npy_links, npy_data)

np.testing.assert_almost_equal(res, torch_rays)

#
# batch_size = 5000
# batch_nb = jnp.ceil(len(ori) / batch_size)
#
# tmp_rgb = []
# for i in range(int(batch_nb - 1)):
#     res = jit_main(ori[i * batch_size: (i + 1) * batch_size],
#                    dir[i * batch_size: (i + 1) * batch_size],
#                    tmin[i * batch_size: (i + 1) * batch_size],
#                    sh_mine[i * batch_size: (i + 1) * batch_size],
#                    max_dt, npy_links, npy_data)
#     res.block_until_ready()
#     tmp_rgb.append(res)
#
# last_dab = len(ori) - (batch_nb - 1) * batch_size
# res = jit_main(ori[int((batch_nb - 1) * batch_size):],
#                dir[int((batch_nb - 1) * batch_size):],
#                tmin[int((batch_nb - 1) * batch_size):],
#                sh_mine[int((batch_nb - 1) * batch_size):],
#                max_dt, npy_links, npy_data)
# tmp_rgb.append(res)
# colors = np.concatenate(tmp_rgb)

complete_colors = np.zeros((rays_origins.shape[0], 3))
complete_colors[mask] = colors

img = complete_colors.reshape([800,800,3])
import cv2
img = (img * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("hello")