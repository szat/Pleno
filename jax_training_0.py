import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import random
from jax.example_libraries import optimizers
from jax import value_and_grad

# Generate some synthetic data for linear regression
key = random.PRNGKey(0)
X = random.normal(key, (100, 1))
Y = 4 * X + 3 #+ 0.1 * random.normal(key, (100, 1))

opt_init, opt_update, get_params = optimizers.sgd(0.001)

# Define the linear regression model
def linear_regression(params, x):
    return jnp.dot(x, params[0]) + params[1]

# Define the loss function (Mean Squared Error)
def mse(params, x, y):
    y_pred = linear_regression(params, x)
    return jnp.mean((y_pred - y)**2)

# Initialize model parameters
# Better for dictionary
params = [jnp.ones((1, 1)), jnp.ones((1,))]
opt_state = opt_init(params)

# Set up the gradient function
grad_mse = grad(mse)

# JIT compile the training loop
def train_step2(inputs, targets, opt_state, opt_update, get_params):
    params = get_params(opt_state)
    loss, gradient = value_and_grad(mse)(params, inputs, targets)
    return loss, opt_update(0, gradient, opt_state) # returns new (updated) optimizer state

learning_rate = 0.01
num_iterations = 2000

# Train the model using JIT-compiled training loop
train_step2_jit = jit(train_step2, static_argnums=(3, 4))
# in train loop:
loss_hist = []
for i in range(num_iterations):
    loss, opt_state = train_step2_jit(X, Y, opt_state, opt_update, get_params)
    loss_hist.append(loss)
import matplotlib.pyplot as plt
plt.plot(loss_hist)
plt.show()
final_params = get_params(opt_state)

# Print the final parameters of the model
print("Final parameters:")
print(final_params)
print(mse(final_params, X, Y))
print(mse(params, X, Y))



import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random

# Generate some synthetic data for linear regression
key = random.PRNGKey(0)
X = random.normal(key, (100, 1))
Y = 4 * X + 3 #+ 0.1 * random.normal(key, (100, 1))

opt_init, opt_update, get_params = optimizers.sgd(0.001)

# Define the linear regression model
def linear_regression(params, x):
    return jnp.dot(x, params[0]) + params[1]

# Define the loss function (Mean Squared Error)
def mse(params, x, y):
    y_pred = linear_regression(params, x)
    return jnp.mean((y_pred - y)**2)

# Initialize model parameters
params = [jnp.ones((1, 1)), jnp.ones((1,))]

# Set up the gradient function
grad_mse = grad(mse)

# JIT compile the training loop
@jit
def train_step(params, X, Y, learning_rate, num_iterations):
    gradient = grad_mse(params, X, Y)
    params = [p - learning_rate * g for p, g in zip(params, gradient)]
    return params

def train(params, X, Y, learning_rate, num_iterations):
    for i in range(num_iterations):
        params = train_step(params, X, Y, learning_rate, num_iterations)
    return params

learning_rate = 0.01
num_iterations = 2000
train(params, X, Y, learning_rate, num_iterations)
