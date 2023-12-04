import torch
import glob
import numpy as np
import os
from .points_utils import get_ini_position_3d
from einops import rearrange
from .base_dy_3d import BaseDataset

class HeartDY3dDataset(BaseDataset):
    def __init__(self, root_dir,img_size,split='train',**kwargs):
        super().__init__(root_dir, split)
        self.root_dir = root_dir
        self.img_size = img_size

        self.read_intrinsics()

        self.read_meta()

    def read_intrinsics(self):
        w = self.img_size[0]
        h = self.img_size[1]
        d = self.img_size[2]
        # w = 160 #224 or 130
        # h = 160 #176 or 110
        # d = 160 #208 or 140
        self.ini_position = get_ini_position_3d(w,h,d)
        self.img_wh = (w, h, d)

    def read_meta(self):

        self.image_paths = glob.glob(os.path.join(self.root_dir, '*'))
        data = np.load(self.image_paths[0])

        self.N_frames = len(data)
        self.imgs = []

        for t in range(self.N_frames):
            img = data[t].astype(np.float32)/255.0 
            img = rearrange(img, 'h w d -> (h w d) 1')
            self.imgs.append(img)
        self.imgs = torch.FloatTensor(np.stack(self.imgs))
