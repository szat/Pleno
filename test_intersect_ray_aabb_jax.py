import numpy as np
import unittest
from scipy.special import *
# from renderer import *
# from spherical_harmonics import *
# from sampling_branch import *
import jax.numpy as jnp
from jax import lax
from jax import grad, jit, vmap
from time import time


def intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max):
    # considers the boundary of the volume as NON intersecting, if tmax <= tmin then NO intersection
    if ray_origin.ndim == 1:
        ray_origin = np.expand_dims(ray_origin, 0)
        ray_inv_dir = np.expand_dims(ray_inv_dir, 0)
    tmin = np.ones(len(ray_origin)) * -np.inf
    tmax = np.ones(len(ray_origin)) * np.inf
    t0 = (box_min - ray_origin) * ray_inv_dir
    t1 = (box_max - ray_origin) * ray_inv_dir
    tsmaller = np.nanmin([t0, t1], axis=0)
    tbigger = np.nanmax([t0, t1], axis=0)
    tmin = np.max([tmin, np.max(tsmaller, axis=1)], axis=0)
    tmax = np.min([tmax, np.min(tbigger, axis=1)], axis=0)
    return tmin, tmax

# just write something
# box_min = jnp.zeros(3)
# box_max = jnp.array([2.0, 2.0, 2.0])

# @jit
def intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max):
    t0 = (box_min - ray_origin) * ray_inv_dir
    t1 = (box_max - ray_origin) * ray_inv_dir
    tsmaller = jnp.nanmin(jnp.vstack([t0, t1]), axis=0)
    tbigger = jnp.nanmax(jnp.vstack([t0, t1]), axis=0)
    tmin = jnp.max(jnp.array([-jnp.inf, jnp.max(tsmaller)]))
    tmax = jnp.min(jnp.array([jnp.inf, jnp.min(tbigger)]))
    return tmin, tmax

class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 3
        self.diff = 1.0
        # So the grid goes like [0.0, 1.0, 2.0]
        self.nb_rays = 5
        self.ex = jnp.ones(3)

    def test_origin_outside_intersect(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, 1.0, 1.0]) #outside of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0]) #point towards cube
        ray_inv_dir = 1/ray_dir
        tnear = 1
        tfar = 3
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_outside_intersect_reverse(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, 1.0, 1.0]) #outside of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0]) #point away from cube
        ray_inv_dir = 1/ray_dir
        tnear = -3
        tfar = -1 #this is not intuitively correct
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_inside_intersect(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([1.0, 1.0, 1.0])  # middle of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0])  # point forward
        ray_inv_dir = 1 / ray_dir
        tnear = -1
        tfar = 1
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_inside_intersect_reverse(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([1.0, 1.0, 1.0])  # middle of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0])  # piont backward
        ray_inv_dir = 1 / ray_dir
        tnear = -1
        tfar = 1
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_on_face_intersect(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([0.0, 1.0, 1.0])  # on face
        ray_dir = jnp.array([1.0, 0.0, 0.0])  # point towards cube
        ray_inv_dir = 1 / ray_dir
        tnear = 0
        tfar = 2
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_on_face_intersect_reverse(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([0.0, 1.0, 1.0])  # on face
        ray_dir = jnp.array([-1.0, 0.0, 0.0])  # point away from cube
        ray_inv_dir = 1 / ray_dir
        tnear = -2
        tfar = 0  # intuitively incorrect
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_not_touching(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([3.0, 3.0, 3.0])  # middle of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf < tn)

        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([3.0, 3.0, 3.0])  # middle of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf < tn)

        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, -1.0, -1.0])  # middle of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf < tn)

        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, -1.0, -1.0])  # middle of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf < tn)

    def test_touching_one_corner(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([1.0, -1.0, 1.0])
        ray_dir = jnp.array([1.0, -1.0, 1.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_bottom(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, 0.0, 0.0])
        ray_dir = jnp.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, 2.0, 2.0])
        ray_dir = jnp.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_bottom_reverse(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([3.0, 0.0, 0.0])  # middle of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top_reverse(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([3.0, 2.0, 2.0])  # middle of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_bottom_minus(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, 0.0, 0.0])  # middle of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top_minus(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, 2.0, 2.0])  # middle of cube
        ray_dir = jnp.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_bottom_reverse_minus(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([3.0, 0.0, 0.0])  # middle of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top_reverse_minus(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([3.0, 2.0, 2.0])  # middle of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1 / ray_dir
        tn, tf = intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_origin_outside_intersect_many_rays(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])
        ray_origin = jnp.array([-1.0, 1.0, 1.0]) #outside of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0]) #point towards cube
        ray_origin = jnp.tile(ray_origin, (1000, 1))
        ray_dir = jnp.tile(ray_dir, (1000, 1))
        ray_inv_dir = 1/ray_dir
        tnear = np.ones(1000)
        tfar = np.ones(1000) * 3

        # t0 = time.time()
        # res_vmap = vmap(fct, in_axes=(None, 0))(input, x)
        t0 = time()
        for i in range(1000):
            intersect_ray_aabb_jax(ray_origin[i], ray_inv_dir[i], box_min, box_max)
        t1 = time()
        print("forloop 1K rays: {}".format(t1 - t0))

        fct = vmap(intersect_ray_aabb_jax, in_axes=(0, 0, None, None))
        t0 = time()
        fct(ray_origin, ray_inv_dir, box_min, box_max)
        t1 = time()
        print("vmap 1K rays: {}".format(t1 - t0))

        fct_jit = jit(fct)
        fct_jit(ray_origin, ray_inv_dir, box_min, box_max)
        t0 = time()
        fct_jit(ray_origin, ray_inv_dir, box_min, box_max)
        t1 = time()
        print("jit vmap 1K rays: {}".format(t1 - t0))

        tn, tf = fct_jit(ray_origin, ray_inv_dir, box_min, box_max)
        # tn, tf = jit(vmap(intersect_ray_aabb_jax))(in_axes=(0, 0, None, None))(ray_origin, ray_inv_dir, box_min, box_max)
        assert len(tn) == 1000
        assert len(tf) == 1000
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_outside_intersect_many_boxes(self):
        box_min = jnp.zeros(3)
        box_max = jnp.array([2.0, 2.0, 2.0])

        box_min = jnp.tile(box_min, (1000, 1))
        box_max = jnp.tile(box_max, (1000, 1))

        ray_origin = jnp.array([-1.0, 1.0, 1.0]) #outside of cube
        ray_dir = jnp.array([1.0, 0.0, 0.0]) #point towards cube
        ray_inv_dir = 1/ray_dir
        tnear = np.ones(1000)
        tfar = np.ones(1000)*3

        t0 = time()
        for i in range(1000):
            intersect_ray_aabb_jax(ray_origin, ray_inv_dir, box_min[i], box_max[i])
        t1 = time()
        print("forloop 1K rays: {}".format(t1 - t0))

        fct = vmap(intersect_ray_aabb_jax, in_axes=(None, None, 0, 0))
        t0 = time()
        fct(ray_origin, ray_inv_dir, box_min, box_max)
        t1 = time()
        print("vmap 1K rays: {}".format(t1 - t0))

        fct_jit = jit(fct)
        fct_jit(ray_origin, ray_inv_dir, box_min, box_max)
        t0 = time()
        fct_jit(ray_origin, ray_inv_dir, box_min, box_max)
        t1 = time()
        print("jit vmap 1K rays: {}".format(t1 - t0))

        tn, tf = fct_jit(ray_origin, ray_inv_dir, box_min, box_max)
        # tn, tf = jit(vmap(intersect_ray_aabb_jax))(in_axes=(0, 0, None, None))(ray_origin, ray_inv_dir, box_min, box_max)
        assert len(tn) == 1000
        assert len(tf) == 1000
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)


if __name__=='__main__':
    unittest.main()
