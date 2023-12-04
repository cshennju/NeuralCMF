import torch
import glob
import numpy as np
import os
from tqdm import tqdm
import json
from .ray_utils import get_ini_position
from einops import rearrange
import collections
from .base_dy import BaseDataset
import imageio


class HeartDYDataset(BaseDataset):
    def __init__(self, root_dir, split='train', **kwargs):
        super().__init__(root_dir, split)
        self.root_dir = root_dir

        self.read_intrinsics()

        self.read_meta()

    def read_intrinsics(self):
        w = 160
        h = 160
        self.ini_position = get_ini_position(h, w)
        self.img_wh = (w, h)

    def read_meta(self):

        self.image_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))

        with open(os.path.join(self.root_dir,"transforms_train.json"), 'r') as f:
            meta = json.load(f,object_pairs_hook=collections.OrderedDict)
        self.poses = []
        for frame in meta['frames']:
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses += [pose]

        self.poses = torch.FloatTensor(self.poses)
        self.N_frames = len(self.image_paths)
        self.imgs = []
        for t in range(self.N_frames):
            img = imageio.imread(self.image_paths[t]).astype(np.float32)/255.0 
            img = rearrange(img, 'h w -> (h w) 1')
            self.imgs.append(img)
        self.imgs = torch.FloatTensor(np.stack(self.imgs)) 
