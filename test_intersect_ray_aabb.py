import numpy as np
import unittest
from scipy.special import *
# from renderer import *
from spherical_harmonics import *
from sampling_branch import *


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.idim = 3
        self.diff = 1.0
        # So the grid goes like [0.0, 1.0, 2.0]
        self.nb_rays = 5

    def test_origin_outside_intersect(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, 1.0, 1.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tnear = 1
        tfar = 3
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_inside_intersect(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([1.0, 1.0, 1.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tnear = -1
        tfar = 1
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_on_face_intersect(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([0.0, 1.0, 1.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tnear = 0
        tfar = 2
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)
