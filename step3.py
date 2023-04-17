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
colors = np.zeros([800*800, 3])
for i in range(len(tmin)):
    tics.append(np.arange(tmin[i], tmax[i], spacing))

for i in range(800*800):
    samples = ori[i, None] + tics[i][:, None] * dir[i, None]
    samples = np.clip(samples, 0, 254)
    sigma = trilinear_interpolation_dot(samples, opacity)
    sigma = np.clip(sigma, a_min=0.0, a_max=100000)
    rgb = trilinear_interpolation_dot(samples, grid)
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

img = colors.reshape([800,800,3])
import cv2
# if nb_sh_channels == 2:
#     img = np.concatenate((img, np.zeros((img_size, img_size, 1)) + 0.5), axis=2)
img = (img * 255).astype(np.uint8)
# if nb_sh_channels == 3:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#
#
# delta_scale = torch.ones_like(delta_scale) / 128
# log_att = -self.opt.step_size * torch.relu(sigma[..., 0]) * delta_scale[
#     good_indices]  # just a scalar if you have only one ray, adrian)
# Ti = torch.exp(log_light_intensity[good_indices])
# weight = Ti * (1.0 - torch.exp(log_att))
# # [B', 3, n_sh_coeffs]
# rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
# rgb = torch.clamp_min(torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5, 0.0, )  # [B', 3]
# rgb = weight[:, None] * rgb[:, :3]
#
# out_rgb[good_indices] += rgb
# log_light_intensity[good_indices] += log_att
# t += self.opt.step_size
#
# # idk why? maybe was just trained that way?
#
# svox_sigma = np.load("/home/adrian/Documents/temp/svox_sigma.npy")
# svox_rgb = np.load("/home/adrian/Documents/temp/svox_rgb.npy")
# svox_sh = np.load("/home/adrian/Documents/temp/svox_sh.npy")
#
# # delta_scale = torch.ones_like(delta_scale) / 128
# # log_att = -self.opt.step_size * torch.relu(sigma[..., 0]) * delta_scale[
# #     good_indices]  # just a scalar if you have only one ray, adrian)
# # Ti = torch.exp(log_light_intensity[good_indices])
# # weight = Ti * (1.0 - torch.exp(log_att))
# # # [B', 3, n_sh_coeffs]
# # rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
# # rgb = torch.clamp_min(torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5, 0.0, )  # [B', 3]
# # rgb = weight[:, None] * rgb[:, :3]
#
# svox_sigma = svox_sigma.clip(min=0.0)
# weights = 1 - np.exp(- svox_sigma * delta_scale * step_size)
#
# np.testing.assert_allclose(weight.numpy(), np.squeeze(weights))
#
# # Ti = np.exp(-np.cumsum(sigma * delta)) first Ti is exp(0) so 1
# svox_rgb = svox_rgb.reshape(-1, 3, 9)
#
# sh_channels = np.einsum("i...,i...,ij...->j...", Ti, weights, interp_sh_coeffs)
#
#
#
