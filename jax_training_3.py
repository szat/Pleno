import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import value_and_grad
from jax.example_libraries import optimizers

# Generate some synthetic data for linear regression
key = random.PRNGKey(0)
params_true = random.normal(key, (9, 1))
params_start = jnp.ones([9,1])
x = random.normal(key, (100, 2))

# simulates the render part to get a color vector
def render(x, params):
    c1 = params[0]*x[0] + params[1]*x[1] + params[2]
    c2 = params[3]*x[0] + params[4]*x[1] + params[5]
    c3 = params[6]*x[0] + params[7]*x[1] + params[8]
    return jnp.array([c1, c2, c3])

render_batch = jit(vmap(render, in_axes=(0, None)))
colors_true_batch = jnp.squeeze(render_batch(x, params_true))

def mse(params, x, colors_true_batch):
    colors_pred_batch = jnp.squeeze(render_batch(x, params))
    out = jnp.mean((colors_pred_batch - colors_true_batch) ** 2)
    return out

grad_mse = value_and_grad(mse, argnums=0)
grad_mse_batch = jit(grad_mse)

opt_init, opt_update, get_params = optimizers.sgd(0.01)
opt_state = opt_init(params_start)

def train_step(x, colors_true_batch, opt_state, opt_update, get_params):
    params = get_params(opt_state)
    loss, gradient = grad_mse_batch(params, x, colors_true_batch)
    return loss, opt_update(0, gradient, opt_state) # returns new (updated) optimizer state

num_iterations = 2000
loss_hist = []
for i in range(num_iterations):
    loss, opt_state = train_step(x, colors_true_batch, opt_state, opt_update, get_params)
    print(i)
    loss_hist.append(loss)

import matplotlib.pyplot as plt
plt.plot(loss_hist)
plt.show()

params_end = np.array(get_params(opt_state))
params_true = np.array(params_true)
np.linalg.norm(params_end - params_true)