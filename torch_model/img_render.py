import copy
import os
import sys
sys.path.append('.')

import cv2
import numpy as np
import torch

from camera import Camera, load_cameras_from_file
from model import RadianceField
from utils import validate_and_find_ray_intersecs


class Renderer:
    def __init__(self,
                 repo_ckpt_path: str,
                 path_to_weights: str,
                 model_name: str,
                 cams_json_path: str,
                 img_size: int = 800,
                 batch_size: int = 1024 * 4,
                 nb_samples: int = 512,
                 nb_sh_channels: int = 3,
                 model_idim: int = 256,
                 device: str = "cuda"):

        self.model_name = model_name
        self.cams_json_path = cams_json_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.nb_samples = nb_samples
        self.nb_sh_channels = nb_sh_channels
        self.model_idim = model_idim
        self.device = device

        if repo_ckpt_path is not None:
            self.model_rf = RadianceField(idim=model_idim, nb_sh_channels=nb_sh_channels, nb_samples=nb_samples,
                                          delta_voxel=2/torch.tensor([model_idim, model_idim, model_idim],
                                                                      dtype=torch.float),
                                          device=device)
            checkpoint = torch.load(repo_ckpt_path)
            self.model_rf.load_state_dict(checkpoint['model_state_dict'])
        else: 
            # Access data arrays using keys
            data = np.load(path_to_svox_weigths, allow_pickle=True)
            npy_radius = data['radius']
            npy_center = data['center']
            npy_links = data['links']
            if model_idim < 512:
                assert 512 % model_idim == 0
                stride = int(512 / model_idim)
                npy_links = npy_links[::stride, ::stride, ::stride] # reduce resolution to half
            npy_density_data = data['density_data']
            npy_sh_data = data['sh_data']
            npy_basis_type = data['basis_type']

            mask = npy_links >= 0
            npy_links = npy_links[mask]

            # hack
            npy_density_data[0] = 0
            npy_sh_data[0] = 0

            density_matrix = np.zeros((model_idim, model_idim, model_idim, 1), dtype=np.float32)
            density_matrix[mask] = npy_density_data[npy_links]
            density_matrix = np.reshape(density_matrix, (model_idim, model_idim, model_idim))
            self.density_matrix = torch.from_numpy(density_matrix)
            self.density_matrix = self.density_matrix.unsqueeze(-1)
            #self.density_data = torch.from_numpy(npy_density_data).to(torch.float32)

            sh_matrix = np.zeros((model_idim, model_idim, model_idim, 27), dtype=np.float16)
            sh_matrix[mask] = npy_sh_data[npy_links]
            sh_matrix = np.reshape(sh_matrix, (model_idim, model_idim, model_idim, 27))
            self.sh_matrix = torch.from_numpy(sh_matrix)
            #self.sh_data = torch.from_numpy(npy_sh_data).to(torch.float16)

            #self.links = torch.from_numpy(npy_links).to(torch.long)
            
            self.model_rf = RadianceField(idim=model_idim, grid=self.sh_matrix, opacity=self.density_matrix, 
                                          nb_sh_channels=nb_sh_channels, nb_samples=nb_samples,
                                          delta_voxel=2/torch.tensor([model_idim, model_idim, model_idim],
                                                                      dtype=torch.float),
                                          device=device)
        self.model_rf.eval()
        self.cameras = load_cameras_from_file(cams_json_path, int(img_size/2), int(img_size/2))
        print("model is loaded!")

    def render_img(self, camera: Camera):
            
        rays_dirs, rays_origins = camera.get_camera_ray_dirs_and_origins()
        rays_origins = rays_origins.astype(np.float32)
        rays_dirs = rays_dirs.astype(np.float32)

        valid_rays_origins, valid_rays_dirs, \
                valid_tmin, valid_tmax, mask = validate_and_find_ray_intersecs(rays_dirs, rays_origins)
        
        # invoke randiance field model:
        torch.cuda.empty_cache()
        rendered_rays, opacity_rays = self.model_rf.render_rays(valid_rays_origins, valid_rays_dirs,
                                                                valid_tmin, valid_tmax, self.batch_size)
        
        # place rendered rays at original positions and calculate img transparency based on 0-density rays:
        rendered_rays = rendered_rays.numpy()
        opacity_rays = opacity_rays.numpy()
        complete_colors = np.ones((rays_origins.shape[0], 3))
        complete_opacities = np.zeros((rays_origins.shape[0]))
        complete_opacities[mask] = opacity_rays
        complete_colors[mask] = rendered_rays
        mask = mask & (complete_opacities > 0)
        complete_colors[~mask] = 1
        complete_colors[complete_colors > 1] = 1
        complete_colors[complete_colors < 0] = 0
        
        # build image:
        img = np.reshape(complete_colors, (self.img_size, self.img_size, self.nb_sh_channels))
        img = np.concatenate((img, np.zeros((self.img_size, self.img_size, 1))), axis=2)
        img[mask.reshape((self.img_size, self.img_size)), -1] = 1
        img = (img * 255).astype(np.uint8)
        if self.nb_sh_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img


if __name__ == "__main__":
    model_name = "lego"
    path_to_svox_weigths = f"/home/diego/data/nerf/ckpt_syn/256_to_512_fasttv/{model_name}/ckpt.npz"
    repo_ckpt_path = f"/home/diego/repos/Pleno/weights/model_7.pt"
    cams_json_path = "/home/diego/data/nerf/nerf_synthetic/nerf_synthetic/lego/transforms_train.json"

    #renderer = Renderer(None, path_to_svox_weigths, model_name, cams_json_path, device="cpu", model_idim=16)
    renderer = Renderer(repo_ckpt_path, None, model_name, cams_json_path, device="cpu", model_idim=16)
    for camera in renderer.cameras:
        print(f"rendering camera {camera.camera_name} ..")
        img = renderer.render_img(camera)
        cv2.imwrite(f"./renders/{model_name}_imgsz{renderer.img_size}_s{renderer.nb_samples}_" + \
                    f"idim{renderer.model_idim}_dev{renderer.device}_{camera.camera_name}.png", img)

