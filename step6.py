import numpy as np
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

colors = np.zeros([800*800, 3])
max_dt = np.max(tmax - tmin)
nb = 500

ori = ori[::1000, :]
dir = dir[::1000, :]
tmin = tmin[::1000]

ori = np.array(ori)
dir = np.array(dir)
tmin = np.array(tmin)
npy_links = np.array(npy_links)
npy_data = np.array(npy_data)

res_non = []
for i in range(len(ori)):
    tics = np.linspace(tmin[i], max_dt + tmin[i], num=nb, dtype=np.float64)
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)
    res_non.append(interp)
res_non = np.stack(res_non, axis=0)




colors_non = []
for i in np.arange(0, 800*800, 100):
    # i = 10
    # x = max_dt + tmin[i] - tmax[i]
    tics = np.linspace(tmin[i], max_dt + tmin[i], num=nb, dtype=np.float64)
    samples = ori[i, None] + tics[:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    interp = trilinear_interpolation_shuffle_zero(samples, npy_links, npy_data)

    # interp_non = []
    # for s in samples:
    #     interp = trilinear_interpolation_shuffle_zero(s, npy_links, npy_data)
    #     interp_non.append(interp)
    # res_non = np.concatenate(interp_non)
    # inter_samples_non.append(interp)

    # sigma = interp[:, :1]
    # rgb = interp[:, 1:]
    #
    # sigma = np.clip(sigma, a_min=0.0, a_max=100000)
    # rgb = rgb.reshape(-1, 3, 9)
    #
    # sh_ray = sh[i][None, None, :]
    # rgb = rgb * sh_ray
    # rgb = np.sum(rgb, axis=2)
    # rgb = rgb + 0.5 #correction 1
    # rgb = np.clip(rgb, a_min=0.0, a_max=100000)
    # tmp = step_size * sigma * delta_scale
    # # tmp = np.clip(tmp, a_min=0.0, a_max=100000)
    # var = 1 - np.exp(-tmp)
    # Ti = np.exp(np.cumsum(-tmp))
    # Ti = Ti[:, None]
    # coefs = Ti * var
    # rgb = coefs * rgb
    # rgb = np.sum(rgb, axis=0)
    # colors[i] = rgb

img = colors.reshape([800,800,3])
import cv2
# if nb_sh_channels == 2:
#     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
img = (img * 255).astype(np.uint8)
# if nb_sh_channels == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
