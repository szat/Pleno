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

xdim, ydim, zdim = npy_links.shape
kernel = np.ones([2, 2, 2]) / 8

from jax import lax

kernel = jnp.ones([2, 2, 2])[:, :, :, jnp.newaxis, jnp.newaxis]

mask_256 = npy_links > 0
mask_256 = jnp.array(mask_256.astype(np.float32))
data = mask_256[None, ::8, ::8, ::8, None]

entry = np.zeros([16, 16, 16])
entry[0, 0, 0] = 1
entry[2, 0, 0] = 1
entry[3, 3, 3] = 1
entry[0, 3, 3] = 1
data = jnp.array(entry.astype(np.float32))
data = data[None, :, :, :, None]

dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))
print(dn)

out = lax.conv_general_dilated(data,    # lhs = image tensor
                               kernel,  # rhs = conv kernel tensor
                               (2,2,2), # window strides
                               'SAME',  # padding mode
                               (1,1,1), # lhs/image dilation
                               (1,1,1), # rhs/kernel dilation
                               dn)      # dimension_numbers
print("out shape: ", out.shape)

mask = out[0, :, :, :, 0] > 0
mask = jnp.array(mask.astype(np.float32))
mask = mask[None, :, :, :, None]

out = lax.conv_general_dilated(mask,    # lhs = image tensor
                               kernel,  # rhs = conv kernel tensor
                               (2,2,2), # window strides
                               'SAME',  # padding mode
                               (1,1,1), # lhs/image dilation
                               (1,1,1), # rhs/kernel dilation
                               dn)      # dimension_numbers
print("out shape: ", out.shape)

def mipmap(mat):
    assert mat.ndim == 3
    xdim, ydim, zdim = mat.shape
    assert xdim == ydim
    assert ydim == zdim

    levels = jnp.log2(xdim)
    mipmap = []

    from jax import lax
    data = jnp.array(mat.astype(jnp.float32))

    occupancy = data > 0
    occupancy = jnp.array(occupancy.astype(jnp.float32))
    mipmap.append(occupancy.astype(int))

    data = data[None, :, :, :, None]
    kernel = jnp.ones([2, 2, 2])[:, :, :, jnp.newaxis, jnp.newaxis]
    dn = lax.conv_dimension_numbers(data.shape, kernel.shape, ('NHWDC', 'HWDIO', 'NHWDC'))

    for i in range(int(levels)):
        out = lax.conv_general_dilated(data,  # lhs = image tensor
                                       kernel,  # rhs = conv kernel tensor
                                       (2, 2, 2),  # window strides
                                       'SAME',  # padding mode
                                       (1, 1, 1),  # lhs/image dilation
                                       (1, 1, 1),  # rhs/kernel dilation
                                       dn)  # dimension_numbers

        occupancy = out > 0
        occupancy = jnp.array(occupancy.astype(jnp.float32))
        data = occupancy
        mipmap.append(occupancy[0, :, :, :, 0].astype(int))

    return mipmap


