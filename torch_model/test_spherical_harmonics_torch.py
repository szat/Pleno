import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from spherical_harmonics import sh_spherical, real_sph_harm_vec_scipy
from torch_model.spherical_harmonics_torch import sh_cartesian
import unittest

import numpy as np
import torch


class TestRenderer(unittest.TestCase):
    def setUp(self):
        pi = np.pi
        self.angles = np.array([[pi/4, pi/4],
                           [pi/2, pi/2],
                           [3*pi/4, 3*pi/4],
                           [5*pi/6, pi],
                            [pi / 6, pi],
                            [pi/4, 7*pi/5],
                            [pi, pi/4], [pi, 5*pi/4], [pi, 2*pi]])

    def spherical_to_cartesian(self, theta, phi):
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.Tensor([x, y, z])

    def test_sh_cartesian_shape_one_entry(self):
        xyz = torch.rand(3)
        res = sh_cartesian(xyz)
        torch.testing.assert_close(res.size(), torch.zeros(9).size())

    def test_sh_cartesian_shape_one_entry_vec(self):
        xyz = torch.rand(1, 3)
        res = sh_cartesian(xyz)
        torch.testing.assert_close(res.size(), torch.zeros(1, 9).size())

    def test_sh_cartesian_shape_many_entry(self):
        xyz = torch.rand(4, 3)
        res = sh_cartesian(xyz)
        torch.testing.assert_close(res.size(), torch.zeros(4, 9).size())

    def test_sh_cartesian_shape_many_entry(self):
        xyz = torch.rand(4, 3)
        res = sh_cartesian(xyz)
        for i in range(4):
            res2 = sh_cartesian(xyz[i, :])
            torch.testing.assert_close(res[i], res2)

    def test_sh_different_formulas(self):
        for i in range(len(self.angles)):
            theta, phi = self.angles[i, 0], self.angles[i, 1]
            res1 = sh_spherical(theta, phi)
            res1 = res1[0]
            res1[np.abs(res1) < 1e-08] = 0
            res1 = torch.from_numpy(res1).type(torch.float32)
            # Note we reverse phi and theta for scipy (idiots)
            res2 = real_sph_harm_vec_scipy(phi, theta)
            res2[np.abs(res2) < 1e-08] = 0
            res2 = torch.from_numpy(res2).type(torch.float32)
            res3 = sh_cartesian(self.spherical_to_cartesian(torch.Tensor([theta]),
                                                            torch.Tensor([phi])))
            res3[torch.abs(res3) < 1e-08] = 0
            torch.testing.assert_close(res1, res2)
            torch.testing.assert_close(res2, res3)


if __name__=='__main__':
    unittest.main()
