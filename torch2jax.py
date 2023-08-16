import numpy as np
from spherical_harmonics import eval_sh_bases_mine
from jax.config import config
config.update("jax_enable_x64", True)

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



# support
canvas = np.zeros([800, 800])
canvas[::10, ::10] = 1
mask = np.zeros([800, 800])
mask[200:-200, 200:-200] = 1
canvas = canvas * mask
canvas = canvas.flatten()
mask = canvas == 1

folder = "/home/adrian/Code/Pleno/data_frames/"
ray_dir = np.load(folder + "ray_dir.npy")
ray_inv = np.load(folder + "ray_inv.npy")
ray_ori = np.load(folder + "ray_ori.npy")
ray_tmin = np.load(folder + "ray_tmin.npy")
ray_tmax = np.load(folder + "ray_tmax.npy")
sh_load = np.load(folder + "_sh.npy")
pytorch_interp_sh = np.load(folder + "_interp_sh_coeff.npy")
pytorch_interp_opacities = np.load(folder + "_interp_opacities.npy")

# from here
samples = np.load(folder+"samples.npy")

sh_mine = eval_sh_bases_mine(ray_dir)
np.testing.assert_almost_equal(sh_mine, sh_load)

from jax import grad, jit, vmap
from jax.config import config
import jax
import jax.numpy as jnp
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

# res_interp = []
samples2 = samples.reshape(-1, 3)
interp = jit_interp(samples2, npy_links, npy_data)

interp_sh_load = np.squeeze(interp_sh_load)
interp_opacities = np.squeeze(interp_opacities)

interp = np.squeeze(interp)

np.testing.assert_almost_equal(interp_opacities[:, None], interp[:, :1])
np.testing.assert_almost_equal(interp_sh_load, interp[:, 1:])


# load the data in the pytorch way
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

mask = npy_links >= 0
npy_links = npy_links[mask]

density_matrix = np.zeros((size_model, size_model, size_model, 1), dtype=np.float32)
density_matrix[mask] = npy_density_data[npy_links]
density_matrix = np.reshape(density_matrix, (size_model, size_model, size_model))

sh_matrix = np.zeros((size_model, size_model, size_model, 27), dtype=np.float16)
sh_matrix[mask] = npy_sh_data[npy_links]
sh_matrix = np.reshape(sh_matrix, (size_model, size_model, size_model, 27))


# from utils import build_samples, eval_sh_bases, trilinear_interpolation
box_min = np.zeros(3)
delta_voxel = np.ones(3)

from trilinear_interpolation import trilinear_interpolation_short

res_interp_sh = trilinear_interpolation_short(samples2, sh_matrix, box_min, delta_voxel)
pytorch_interp_sh = np.squeeze(pytorch_interp_sh)
np.testing.assert_almost_equal(pytorch_interp_sh, res_interp_sh)

tmp = trilinear_interpolation_to_vmap(samples2[0, :], npy_links, npy_sh_data)

jax_interp_sh = jit_interp(samples2, npy_links, npy_sh_data)
jax_interp_sh = np.squeeze(jax_interp_sh)

np.testing.assert_almost_equal(pytorch_interp_sh, jax_interp_sh)