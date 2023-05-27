
import argparse


class RadianceFieldOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Radience Field options")

        # paths:
        self.parser.add_argument("--out_weights_path",
                                 type=str,
                                 help="path to output folder for weights",
                                 default="/home/diego/repos/Pleno/weights/")
        self.parser.add_argument("--cams_json_path",
                                 type=str,
                                 help="path to input json file for camera rotations",
                                 default="/home/diego/data/nerf/nerf_synthetic/nerf_synthetic/lego/transforms_train.json")
        self.parser.add_argument("--imgs_folder_path",
                                 type=str,
                                 help="path to input images for camera rotations",
                                 default="/home/diego/data/nerf/nerf_synthetic/nerf_synthetic/lego/train/")
        self.parser.add_argument("--img_ext",
                                 type=str,
                                 help="image file extension",
                                 default="png")

        # training options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=2048*64)
        self.parser.add_argument("--nb_samples",
                                 type=int,
                                 help="numer of samples for ray marching",
                                 default=128)
        self.parser.add_argument("--max_epochs",
                                 type=int,
                                 help="max. train epochs",
                                 default=100)
        self.parser.add_argument("--device",
                                 type=str,
                                 help="torch device name to load model",
                                 default="cuda")
        self.parser.add_argument("--idim",
                                 type=int,
                                 help="resolution of model",
                                 default=16)
        self.parser.add_argument("--img_width",
                                 type=int,
                                 help="target image width",
                                 default=800)
        self.parser.add_argument("--img_height",
                                 type=int,
                                 help="target image height",
                                 default=800)
        # optimizer options:
        self.parser.add_argument("--opt_lr",
                                 type=float,
                                 help="optimizer lr",
                                 default=0.05)
        self.parser.add_argument("--opt_momentum",
                                 type=float,
                                 help="optimizer momentum",
                                 default=0.99)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
