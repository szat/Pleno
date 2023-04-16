import math

import numpy as np


class Camera:

    def __init__(self,
                 camera_angle_x: float,
                 transform_matrix: np.ndarray,
                 cx: float = 400,
                 cy: float = 400,
                 size_model: int = 256,
                ):
        # center of object to render is at position (0, 0, 0)
        # its dimensions are [1,1,1]
        self.transform_matrix = transform_matrix
        self.transform_matrix[:3, 3] = (self.transform_matrix[:3, 3]  + 1)*size_model/2
        self.transform_matrix[3, 3] = 1
        self.origin = np.array([0, 0, 0, 1]) @ self.transform_matrix.T
        self.origin = self.origin[:3]
        self.cx = cx
        self.cy = cy
        fx = cx / math.tan(camera_angle_x)
        fy = cy / math.tan(camera_angle_x)
        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_camera_vectors(self):
        x = np.array([1, 0, 0, 0]) @ self.transform_matrix.T
        y = np.array([0, 1, 0, 0]) @ self.transform_matrix.T
        z = np.array([0, 0, 1, 0]) @ self.transform_matrix.T
        return x[:3], -y[:3], -z[:3]

    def get_camera_ray_dirs_and_origins(self):
        x, y, z = self.get_camera_vectors()
        tics_x = np.expand_dims(np.linspace(-1, 1, self.cx*2), 1)
        tics_y = np.expand_dims(np.linspace(-1, 1, self.cy*2), 1)

        xx = tics_x * x
        yy = tics_y * y
        xx = np.expand_dims(xx, 0)
        yy = np.expand_dims(yy, 1)
        rays = xx + yy
        zz = np.expand_dims(z, [0, 1])
        rays_dirs = rays + zz
        rays_dirs = rays_dirs / np.expand_dims(np.linalg.norm(rays_dirs, axis=2), 2)
        rays_origins = np.tile(self.origin, (self.cx*2 * self.cy*2, 1))
        rays_dirs = rays_dirs.reshape((-1, 3))
        return rays_dirs, rays_origins
