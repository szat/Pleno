import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random

# Generate some synthetic data for linear regression
key = random.PRNGKey(0)
x = random.normal(key, (100, 1))
params_start = jnp.array([1.0, 1.0]) #[jnp.ones((1, 1)), jnp.ones((1,))]
params_true = jnp.array([4.0, 3.0])

# Define the linear regression model
def linear_regression(params, x):
    return jnp.dot(x, params[0]) + params[1]

y_true = linear_regression(params_true, x)

# Define the loss function (Mean Squared Error)
def mse(params, x, y_true):
    y_pred = linear_regression(params, x)
    return jnp.mean((y_pred - y_true)**2)

grad_mse = grad(mse)

# JIT compile the training loop
def train_step(params, x, y, learning_rate):
    gradient = grad_mse(params, x, y)
    params = [p - learning_rate * g for p, g in zip(params, gradient)]
    return params

train_step_jit = jit(train_step, static_argnums=(3))

def train(params, x, y, learning_rate, num_iterations):
    for i in range(num_iterations):
        params = train_step_jit(params, x, y, learning_rate)
    return params

learning_rate = 0.01
num_iterations = 2000
params_end = train(params_start, x, y_true, learning_rate, num_iterations)

