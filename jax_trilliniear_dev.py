import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from trilinear_interpolation import *

nb_samples = 100
dim = 10
total_size = 100*100*100
nb_non_zero = 500000

grid = np.zeros(total_size)
idx = np.random.choice(total_size, nb_non_zero, replace=False)
links = np.random.choice(nb_non_zero, nb_non_zero, replace=False)
grid[idx] = links
grid = grid.reshape(100, 100, 100).astype(int)
vecs = np.random.uniform(2, 98, [nb_samples, 3])
data = np.random.rand(nb_non_zero, dim)
data[0] = 0

res_np = trilinear_interpolation_shuffle_zero(vecs, grid, data)

vecs = jnp.array(vecs)
grid = jnp.array(grid)
data = jnp.array(data)

res_jax = []
for v in vecs:
    res_jax.append(trilinear_jax(v, grid, data))

res_vmap = vmap(trilinear_jax, in_axes=(0, None, None))(vecs, grid, data)


res_jax = jnp.concatenate(res_jax)
res_jax = res_jax.reshape(res_vmap.shape)
np.testing.assert_allclose(res_jax, res_vmap)
np.testing.assert_allclose(res_np, res_vmap)

