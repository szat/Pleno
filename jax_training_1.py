import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random

# Generate some synthetic data for linear regression
key = random.PRNGKey(0)
X = random.normal(key, (100, 1))
Y = 4 * X + 3 #+ 0.1 * random.normal(key, (100, 1))

# Define the linear regression model
def linear_regression(params, x):
    return jnp.dot(x, params[0]) + params[1]

# Define the loss function (Mean Squared Error)
def mse(params, x, y):
    y_pred = linear_regression(params, x)
    return jnp.mean((y_pred - y)**2)

def mse_2(params, rays, colors):
    colors_pred = render(params, rays)
    return jnp.mean((colors - colors_pred)**2)

grad_mse_2 = grad(mse_2())

# Initialize model parameters
params = [jnp.ones((1, 1)), jnp.ones((1,))]

# Set up the gradient function
grad_mse = grad(mse)

# JIT compile the training loop
def train_step(params, X, Y, learning_rate, num_iterations):
    gradient = grad_mse(params, X, Y)
    params = [p - learning_rate * g for p, g in zip(params, gradient)]
    return params

train_step_jit = jit(train_step, static_argnums=(3, 4))

def train(params, X, Y, learning_rate, num_iterations):
    for i in range(num_iterations):
        params = train_step_jit(params, X, Y, learning_rate, num_iterations)
    return params

learning_rate = 0.01
num_iterations = 2000
train(params, X, Y, learning_rate, num_iterations)
