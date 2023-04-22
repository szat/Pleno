from typing import List
import math
import json

import numpy as np


class Camera:

    def __init__(self,
                 camera_angle_x: float,
                 transform_matrix: np.ndarray,
                 camera_name: str,
                 img_path: str,
                 cx: float = 400,
                 cy: float = 400,
                ):
        # object model is in the cube with corners (-1,-1,-1) and (1,1,1)
        self.camera_name = camera_name
        self.img_path = img_path
        self.transform_matrix = np.array(transform_matrix)
        self.origin = np.array([0, 0, 0, 1]) @ self.transform_matrix.T
        self.origin = self.origin[:3]
        self.cx = cx
        self.cy = cy
        self.fx = self.cx / math.tan(camera_angle_x / 2)
        self.fy = self.cy / math.tan(camera_angle_x / 2)

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
        xx = (np.expand_dims(xx, 0) * self.cx + 0.5) / self.fx
        yy = (np.expand_dims(yy, 1) * self.cy + 0.5) / self.fy
        rays = xx + yy
        zz = np.expand_dims(z, [0, 1])
        rays_dirs = rays + zz
        rays_dirs = rays_dirs / np.expand_dims(np.linalg.norm(rays_dirs, axis=2), 2)
        rays_origins = np.tile(self.origin, (self.cx*2 * self.cy*2, 1))
        rays_dirs = rays_dirs.reshape((-1, 3))
        return rays_dirs, rays_origins


def load_cams_json(path: str):
    with open(path) as f:
        json_obj = json.load(f)
        return json_obj


def load_cameras_from_file(json_path: str, width: int, height: int) -> List[Camera]:
    json_file = load_cams_json(json_path)
    camera_angle_x = json_file["camera_angle_x"]
    rot_list = json_file["frames"]
    cameras = []
    for rot in rot_list:
        img_path = rot["file_path"]
        camera_name = img_path.split("/")[2]
        cameras.append(Camera(camera_angle_x, rot["transform_matrix"], camera_name, img_path,
                              width, height))
    return cameras
