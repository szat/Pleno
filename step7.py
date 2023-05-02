import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
from jax import device_put
import jax
import time
import contextlib
import timeit

from time import time


def timer_func(func):
    import numpy
    def wrap_func(*args, **kwargs):
        times = np.zeros(30)
        for i in range(30):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            times[i] = t2-t1
        print(f'Function {func.__name__!r} executed in {times.mean()*1000:.4f}ms, std: {times.std()*1000:.4f}ms')
        return result
    return wrap_func

#define random link grid with integer values going from negative to n.

nb_links = 16*16*16
nb_non_neg = 1000
links_idx = np.arange(nb_links)
idx_in_links = np.random.choice(links_idx, size=nb_non_neg, replace=False)
values_idx = np.arange(nb_non_neg).astype(int)
links = np.ones([16, 16, 16]) * -1
links = links.reshape(-1)
links[idx_in_links] = values_idx
links = links.reshape([16, 16, 16])
values_compressed = np.random.rand(nb_non_neg)
vecs = np.random.rand(100, 3)
vecs = vecs * 15

# now we have all of our values, lets send them all to the gpu via jax and see what happens
links = jnp.array(links)
values_compressed = jnp.array(values_compressed)
vecs = jnp.array

# print(links.device_buffer.device()) (vecs)
xyz_floor = jnp.floor(vecs)
diff = vecs - xyz_floor
x0, y0, z0 = xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]
x0 = jnp.int32(x0)
y0 = jnp.int32(y0)
z0 = jnp.int32(z0)
l000 = links[x0, y0, z0]

key = random.PRNGKey(0)
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
@timer_func
def example2():
    np.dot(x, x.T)
example2()


x = device_put(x)
@timer_func
def example1():
    jnp.dot(x, x.T).block_until_ready()
example1()


x = random.normal(key, (1000000,))

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

@timer_func
def example3():
    selu(x).block_until_ready()
example3()

selu_jit = jit(selu)
@timer_func
def example4():
    selu_jit(x).block_until_ready()
example4()

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
  return jnp.dot(mat, v)


x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
with time_measure('execution time'):
    jnp.dot(x, x.T).block_until_ready()

