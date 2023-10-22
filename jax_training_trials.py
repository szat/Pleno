import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt


# Generate synthetic data
key = random.PRNGKey(0)
x = random.normal(key, (10, 5))
y = jnp.dot(x, jnp.array([1., 2., 3., 4., 5.])) + 0.1 * random.normal(key, (10,))


LEARNING_RATE = 1e-4
# Get the optimizer objects
opt_init, opt_update, get_params = optimizers.adam(step_size=LEARNING_RATE)

# Define the loss function
def mse(params, x, y):
    preds = jnp.dot(x, params)
    return jnp.mean((preds - y) ** 2)

# Define the update function
def update(params, x, y):
    grads = grad(mse)(params, x, y)
    return opt_update(0, grads, params)

# Define the training loop
def train_loop(x, y):
    opt_init, opt_update, get_params = optimizers.sgd(step_size=0.1)
    params = jnp.zeros(5)
    opt_state = opt_init(params)
    for i in range(500):
        params = get_params(opt_state)
        params = update(params, x, y)
        opt_state = opt_update(0, grads(params), opt_state)
        if i % 10 == 0:
            loss = mse(params, x, y)
            print(f"Step {i}, loss {loss:.4f}")
    return params

# Train the model
params = train_loop(x, y)

print(f"\nLearned parameters: {params}")




import jax
import jax.numpy as jnp

def synthetic_data():
    x = jnp.linspace(-1, 1, 100)
    y = 2 * x + 1 + jax.random.normal(jax.random.PRNGKey(0), (100,))
    return x, y

def l2_loss(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

def train_step(optimizer, params, x, y):
    grads = jax.grad(l2_loss)(params, x, y)
    return optimizer.update(grads, params)

def train_loop(model):
    optimizer = optax.sgd(learning_rate=0.01)
    params = model.init(jax.random.PRNGKey(0), synthetic_data()[0])
    for i in range(1000):
        x, y = synthetic_data()
        params = train_step(optimizer, params, x, y)
        if i % 100 == 0:
            print(f"Loss at step {i}: {l2_loss(params, x, y)}")



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
# Better for dictionary
params = [jnp.ones((1, 1)), jnp.ones((1,))]
opt_state = opt_init(params)

# Set up the gradient function
grad_mse = grad(mse)

# JIT compile the training loop
@jit
def train_step(params, X, Y, learning_rate, num_iterations):
    gradient = grad_mse(params, X, Y)
    params = [p - learning_rate * g for p, g in zip(params, gradient)]
    return params

from jax import value_and_grad
def train_step2(inputs, targets, opt_state, opt_update, get_params):
    params = get_params(opt_state)
    loss, gradient = value_and_grad(mse)(params, inputs, targets)
    return loss, opt_update(0, gradient, opt_state) # returns new (updated) optimizer state


def train(params, X, Y, learning_rate, num_iterations):
    for i in range(num_iterations):
        params = train_step(params, X, Y, learning_rate, num_iterations)
    return params

learning_rate = 0.01
num_iterations = 2000

# Train the model using JIT-compiled training loop
# params = train(params, X, Y, learning_rate, num_iterations)

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