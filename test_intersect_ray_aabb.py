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
        ray_origin = np.array([-1.0, 1.0, 1.0]) #outside of cube
        ray_dir = np.array([1.0, 0.0, 0.0]) #point towards cube
        ray_inv_dir = 1/ray_dir
        tnear = 1
        tfar = 3
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_outside_intersect_reverse(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, 1.0, 1.0]) #outside of cube
        ray_dir = np.array([-1.0, 0.0, 0.0]) #point away from cube
        ray_inv_dir = 1/ray_dir
        tnear = -3
        tfar = -1 #this is not intuitively correct
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_outside_intersect_many(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, 1.0, 1.0]) #outside of cube
        ray_dir = np.array([1.0, 0.0, 0.0]) #point towards cube
        ray_origin = np.tile(ray_origin, (2, 1))
        ray_dir = np.tile(ray_dir, (2, 1))
        ray_inv_dir = 1/ray_dir
        tnear = np.array([1, 1])
        tfar = np.array([3, 3])
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert len(tn) == 2
        assert len(tf) == 2
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_inside_intersect(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([1.0, 1.0, 1.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0]) #point forward
        ray_inv_dir = 1/ray_dir
        tnear = -1
        tfar = 1
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_inside_intersect_reverse(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([1.0, 1.0, 1.0]) #middle of cube
        ray_dir = np.array([-1.0, 0.0, 0.0]) #piont backward
        ray_inv_dir = 1/ray_dir
        tnear = -1
        tfar = 1
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_on_face_intersect(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([0.0, 1.0, 1.0]) #on face
        ray_dir = np.array([1.0, 0.0, 0.0]) #point towards cube
        ray_inv_dir = 1/ray_dir
        tnear = 0
        tfar = 2
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)

    def test_origin_on_face_intersect_reverse(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([0.0, 1.0, 1.0]) #on face
        ray_dir = np.array([-1.0, 0.0, 0.0]) #point away from cube
        ray_inv_dir = 1/ray_dir
        tnear = -2
        tfar = 0 #intuitively incorrect
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        np.testing.assert_almost_equal(tnear, tn)
        np.testing.assert_almost_equal(tfar, tf)


    def test_not_touching(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([3.0, 3.0, 3.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert(tf < tn)

        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([3.0, 3.0, 3.0]) #middle of cube
        ray_dir = np.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert(tf < tn)

        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, -1.0, -1.0]) #middle of cube
        ray_dir = np.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert(tf < tn)

        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, -1.0, -1.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert(tf < tn)

    def test_touching_one_corner(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([1.0, -1.0, 1.0])
        ray_dir = np.array([1.0, -1.0, 1.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert(tf <= tn)

    def test_along_edge_bottom(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, 0.0, 0.0])
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, 2.0, 2.0])
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_bottom_reverse(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([3.0, 0.0, 0.0]) #middle of cube
        ray_dir = np.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top_reverse(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([3.0, 2.0, 2.0]) #middle of cube
        ray_dir = np.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_bottom_minus(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, 0.0, 0.0]) #middle of cube
        ray_dir = np.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top_minus(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([-1.0, 2.0, 2.0]) #middle of cube
        ray_dir = np.array([-1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_bottom_reverse_minus(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([3.0, 0.0, 0.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)

    def test_along_edge_top_reverse_minus(self):
        box_min = np.zeros(3)
        box_max = np.array([2.0, 2.0, 2.0])
        ray_origin = np.array([3.0, 2.0, 2.0]) #middle of cube
        ray_dir = np.array([1.0, 0.0, 0.0])
        ray_inv_dir = 1/ray_dir
        tn, tf = intersect_ray_aabb(ray_origin, ray_inv_dir, box_min, box_max)
        assert (tf <= tn)


if __name__=='__main__':
    unittest.main()
