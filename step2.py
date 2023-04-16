import numpy as np

svox_pos = np.load("/home/adrian/Documents/temp/svox_pos.npy")
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

xyz = svox_pos
# xyz = xyz - origin
# xyz = xyz / np.array([dx, dy, dz])  # , dtype=float), axis=1)

xyz_floor = np.floor(xyz)
# np.sum(np.abs(xyz_floor - l.numpy()))
diff = xyz - xyz_floor

xd, yd, zd = diff[:, 0], diff[:, 1], diff[:, 2]

xyz_floor = xyz_floor.astype(int)
x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]

x_zeros = np.zeros(x0.shape, dtype=int)
x_zeros[xd >= 0] += 1
x1 = x0 + x_zeros

y_zeros = np.zeros(y0.shape, dtype=int)
y_zeros[yd >= 0] += 1
y1 = y0 + y_zeros

z_zeros = np.zeros(z0.shape, dtype=int)
z_zeros[zd >= 0] += 1
z1 = z0 + z_zeros

np.testing.assert_allclose(grid[x0, y0, z0], rgb000.numpy())
np.testing.assert_allclose(grid[x1, y0, z0], rgb100.numpy())
np.testing.assert_allclose(grid[x0, y1, z0], rgb010.numpy())
np.testing.assert_allclose(grid[x1, y1, z0], rgb110.numpy())

np.testing.assert_allclose(grid[x0, y0, z1], rgb001.numpy())
np.testing.assert_allclose(grid[x1, y0, z1], rgb101.numpy())
np.testing.assert_allclose(grid[x0, y1, z1], rgb011.numpy())
np.testing.assert_allclose(grid[x1, y1, z1], rgb111.numpy())

np.testing.assert_allclose(opacity[x0, y0, z0], sigma000.numpy())
np.testing.assert_allclose(opacity[x1, y0, z0], sigma100.numpy())
np.testing.assert_allclose(opacity[x0, y1, z0], sigma010.numpy())
np.testing.assert_allclose(opacity[x1, y1, z0], sigma110.numpy())

np.testing.assert_allclose(opacity[x0, y0, z1], sigma001.numpy())
np.testing.assert_allclose(opacity[x1, y0, z1], sigma101.numpy())
np.testing.assert_allclose(opacity[x0, y1, z1], sigma011.numpy())
np.testing.assert_allclose(opacity[x1, y1, z1], sigma111.numpy())

from trilinear_interpolation import *

# tmp = trilinear_interpolation(svox_pos, opacity)
avec = trilinear_coefficients(svox_pos).T

np.testing.assert_allclose(avec, svox_avec.numpy())

cvec = np.hstack([opacity[x0, y0, z0],
    opacity[x0, y0, z1],
    opacity[x0, y1, z0],
    opacity[x0, y1, z1],
    opacity[x1, y0, z0],
    opacity[x1, y0, z1],
    opacity[x1, y1, z0],
    opacity[x1, y1, z1]])

res = np.expand_dims(np.einsum('ij,ij...->i...', avec, cvec), 1)

np.testing.assert_allclose(res, sigma.numpy())
# ok

res2 = trilinear_interpolation_dot(pos.)


cvec = np.zeros([5000, 8, 27])
cvec[:, 0, :] = grid[x0, y0, z0]
cvec[:, 1, :] = grid[x0, y0, z1]
cvec[:, 2, :] = grid[x0, y1, z0]
cvec[:, 3, :] = grid[x0, y1, z1]
cvec[:, 4, :] = grid[x1, y0, z0]
cvec[:, 5, :] = grid[x1, y0, z1]
cvec[:, 6, :] = grid[x1, y1, z0]
cvec[:, 7, :] = grid[x1, y1, z1]

res = np.einsum('ij,ijk->ik', avec, cvec)
np.testing.assert_allclose(res, rgb.numpy())

res2 = trilinear_interpolation_dot(pos.numpy(), opacity)

np.testing.assert_allclose(res, res2)
np.testing.assert_allclose(sigma.numpy(), res2)