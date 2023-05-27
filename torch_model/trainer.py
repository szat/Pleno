
from datetime import datetime
import os

from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import torch

from camera import Camera, load_cameras_from_file
from model import RadianceField
from utils import validate_and_find_ray_intersecs


class Trainer:
    def __init__(self, options):
        self.opts = options
        self.cameras = load_cameras_from_file(self.opts.cams_json_path,
                                              int(self.opts.img_width/2), int(self.opts.img_height/2))
        self.ray_dirs, self.ray_origins = [], []
        self.ray_tmin, self.ray_tmax = [], []
        self.rgb_colors = []
        total_nb_rays = 0
        for cam in self.cameras:
            img_path = os.path.join(self.opts.imgs_folder_path, cam.camera_name + "." + self.opts.img_ext)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # keep alpha channel
            assert not img is None
            if img.shape[:2] != (self.opts.img_height, self.opts.img_width):
                img = cv2.resize(img, (self.opts.img_width, self.opts.img_height))
            img = np.reshape(img, (-1, 4))
            cam_ray_dirs, cam_ray_origins = cam.get_camera_ray_dirs_and_origins()
            cam_ray_dirs = cam_ray_dirs.astype(np.float32)
            cam_ray_origins = cam_ray_origins.astype(np.float32)
            mask = img[:, 3] > 0

            valid_rays_origins, valid_rays_dirs, \
                    valid_tmin, valid_tmax, _ = validate_and_find_ray_intersecs(cam_ray_dirs[mask],
                                                                                   cam_ray_origins[mask])
            self.ray_dirs.append(valid_rays_dirs)
            self.ray_origins.append(valid_rays_origins)
            self.ray_tmin.append(valid_tmin)
            self.ray_tmax.append(valid_tmax)
            self.rgb_colors.append(img[mask, 0:3])
            total_nb_rays += valid_rays_dirs.shape[0]

        print(f"total number of train rays {total_nb_rays}")
        suffle_idxs = torch.randperm(total_nb_rays)
        self.ray_dirs = torch.Tensor(np.concatenate(self.ray_dirs))[suffle_idxs]
        self.ray_origins = torch.Tensor(np.concatenate(self.ray_origins))[suffle_idxs]
        self.ray_tmin = torch.Tensor(np.concatenate(self.ray_tmin))[suffle_idxs]
        self.ray_tmax = torch.Tensor(np.concatenate(self.ray_tmax))[suffle_idxs]
        self.rgb_colors = torch.Tensor(np.concatenate(self.rgb_colors))[suffle_idxs]
        self.radiance_field = RadianceField(idim=self.opts.idim, nb_sh_channels=3,
                                            nb_samples=self.opts.nb_samples,
                                            opt_lr=self.opts.opt_lr, opt_momentum=self.opts.opt_momentum,
                                            device=self.opts.device)
        
    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, batch_start in enumerate(range(0, self.ray_dirs.shape[0], self.opts.batch_size)):
            batch_end = min(batch_start + self.opts.batch_size, self.ray_dirs.shape[0])
            origins_batched = self.ray_origins[batch_start:batch_end]
            origins_batched = origins_batched.to(self.opts.device)
            dirs_batched = self.ray_dirs[batch_start:batch_end]
            dirs_batched = dirs_batched.to(self.opts.device)
            tmin_batched = self.ray_tmin[batch_start:batch_end]
            tmin_batched = tmin_batched.to(self.opts.device)
            tmax_batched = self.ray_tmax[batch_start:batch_end]
            tmax_batched = tmax_batched.to(self.opts.device)
            colors_batched = self.rgb_colors[batch_start:batch_end]
            colors_batched = colors_batched.to(self.opts.device)
            loss = self.radiance_field.train_step(origins_batched, dirs_batched,
                                                  tmin_batched, tmax_batched,
                                                  colors_batched)
            # Gather data and report
            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * self.ray_dirs.shape[0]/self.opts.batch_size + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
        
    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0
        for epoch in range(self.opts.max_epochs):
            print('EPOCH {}:'.format(epoch_number + 1))
            self.radiance_field.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer)
            writer.flush()

            torch.save({
                'epoch': epoch_number,
                'model_state_dict': self.radiance_field.state_dict(),
                'optimizer_state_dict': self.radiance_field.optimizer.state_dict(),
                'loss': avg_loss,
                }, os.path.join(self.opts.out_weights_path, f"model_{epoch_number}.pt"))

            epoch_number += 1
