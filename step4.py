import numpy as np
from trilinear_interpolation import *
from sampling_branch import intersect_ray_aabb
from spherical_harmonics import eval_sh_bases_mine
# smomethn
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

mask = npy_links >= 0
grid = np.zeros([256, 256, 256, 27])
grid[mask] = npy_sh_data[npy_links[mask]]
opacity = np.zeros([256, 256, 256, 1])
opacity[mask] = npy_density_data[npy_links[mask]]

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
tics = []
for i in range(len(tmin)):
    tics.append(np.arange(tmin[i], tmax[i], spacing))

tics_len = np.fromiter((len(x) for x in tics), int)
# tics_idx = np.cumsum(tics_len)
# tics_idx = np.insert(tics_idx, 0, 0, axis=0)

colors = np.zeros([0, 3])

total_size = 640000
chunk_size = 1000
nb_iter = total_size / chunk_size
nb_extra = total_size - int(nb_iter) * chunk_size
nb_iter = int(nb_iter)

for iter in range(nb_iter):
    ray_idx_start = iter * chunk_size
    ray_idx_end = (iter+1) * chunk_size
    nb_samples = np.sum(tics_len[ray_idx_start:ray_idx_end])
    samples = np.zeros([nb_samples, 3])

    start = 0
    end = tics_len[ray_idx_start]
    for i in range(chunk_size):
        samples[start:end] = ori[ray_idx_start+i, None] + tics[ray_idx_start+i][:, None] * dir[ray_idx_start+i, None]
        start = end
        end += tics_len[ray_idx_start+i+1]

    samples[samples < 0] = 0
    samples[samples > 254] = 254
    sigma = trilinear_interpolation_dot(samples, opacity)
    sigma[sigma < 0] = 0
    rgb = trilinear_interpolation_dot(samples, grid)
    rgb = rgb.reshape(-1, 3, 9)

    start = 0
    end = tics_len[ray_idx_start]
    for i in range(chunk_size):
        sh_ray = sh[ray_idx_start+i][None, None, :]
        rgb[start:end] = rgb[start:end, :, :] * sh_ray
        start = end
        end += tics_len[ray_idx_start+i+1]

    rgb = np.sum(rgb, axis=2)
    rgb = rgb + 0.5 #correction 1
    rgb[rgb < 0] = 0.0
    tmp = step_size * sigma * delta_scale
    var = 1 - np.exp(-tmp)
    Ti = np.exp(np.cumsum(-tmp))
    Ti = Ti[:, None]
    coefs = Ti * var
    rgb = coefs * rgb
    rgb = np.sum(rgb, axis=0)

    start = 0
    end = tics_len[ray_idx_start]
    new_colors = np.ones([chunk_size, 3])
    for i in range(chunk_size):
        new_colors[i] = np.sum(rgb[start:end], axis=0)
        start = end
        end += tics_len[ray_idx_start+i+1]

    colors = np.vstack([colors, new_colors])

    print(f"finished iter {iter}")


img = colors.reshape([800,800,3])
import cv2
img = (img * 255).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


