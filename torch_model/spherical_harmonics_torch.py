import math

import torch


def sh_cartesian(xyz: torch.Tensor, K: torch.Tensor):
    if xyz.ndim == 1:
        r = torch.linalg.norm(xyz)
        xyz = xyz / r
        x, y, z = xyz[0], xyz[1], xyz[2]
        vec = torch.Tensor([1, y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2], device=xyz.device)
        return vec * K
    else:
        r = torch.linalg.norm(xyz, axis=1)
        r = torch.unsqueeze(r, 1)
        xyz = xyz / r
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ones = torch.ones(x.size(), device=xyz.device)
        vec = torch.vstack([ones, y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2]).T
        return vec * K
