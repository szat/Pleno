#
import numpy as np

svox_ori = np.load("/home/adrian/Documents/temp/svox_ori.npy")
svox_dir = np.load("/home/adrian/Documents/temp/svox_dir.npy")
svox_ori = svox_ori[:5000, :]
svox_dir = svox_dir[:5000, :]

# origins = self.world2grid(rays.origins)
# dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
# viewdirs = dirs
# B = dirs.size(0)
# assert origins.size(0) == B
# gsz = self._grid_size()
# dirs = dirs * (self._scaling * gsz).to(device=dirs.device)
# delta_scale = 1.0 / dirs.norm(dim=1)
# dirs *= delta_scale.unsqueeze(-1)  # negates the effect of the last 4 lines, the directions are still the same

# from svox.py line 1494
gsz = 256
offset = 0.5 * gsz - 0.5  # 0.5 * 256 - 0.5
scaling = 0.5 * gsz  # 0.5 * 256
# offset + points * scaling
# load data of rays
svox_ori = offset + svox_ori * scaling

np.testing.assert_allclose(svox_ori, origins.numpy())

from sampling_branch import intersect_ray_aabb
spacing = 0.5
box_top = np.ones(3)*256
box_bottom = np.zeros(3)
# to compute the tmin and tmax, they do a offset of -0.5 first, so add -0.5 to box_min and box_max
svox_dir = svox_dir / np.expand_dims(np.linalg.norm(svox_dir, axis=1), 1)
svox_inv = 1/svox_dir
tmin_, tmax_ = intersect_ray_aabb(svox_ori, svox_inv, box_bottom-0.5, box_top-0.5)

svox_tmin = np.load("/home/adrian/Documents/temp/svox_tmin.npy")
svox_tmax = np.load("/home/adrian/Documents/temp/svox_tmax.npy")

# np.testing.assert_allclose(svox_tmin, tmin_)
# np.testing.assert_allclose(svox_tmax, tmax_)
np.testing.assert_allclose(t.numpy(), tmin_)
np.testing.assert_allclose(tmax.numpy(), tmax_)
# good enough

tmin_ = svox_tmin
tmax_ = svox_tmax

mask = tmin_ < tmax_
svox_ori = svox_ori[mask]
svox_dir = svox_dir[mask]
tmin_ = tmin_[mask]
tmax_ = tmax_[mask]
tics = []
for i in range(len(tmin_)):
    tics.append(np.arange(tmin_[i], tmax_[i], spacing))

iter = 0
sample_points_5000 = np.zeros([5000, 3])
for i in range(5000):
    # t = tics[i]
    ori = svox_ori[i]
    dir = svox_dir[i]
    ori = np.expand_dims(ori, axis=1)
    dir = np.expand_dims(dir, axis=1)
    t = tics[i]
    samples = (ori + t[iter] * dir).T
    ori = ori.T
    dir = dir.T
    sample_points = samples
    sample_points = np.clip(sample_points, 0, 254) # svox2.py, line 717
    sample_points_5000[i] = sample_points

svox_pos = np.load("/home/adrian/Documents/temp/svox_pos.npy")
np.testing.assert_allclose(pos.numpy(), svox_pos)

# this is ok
np.testing.assert_allclose(sample_points_5000, pos.numpy())

