from torch.utils.data import Dataset
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return 90

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            img_idxs = np.random.choice(len(self.imgs), self.batch_size)
            time_idxs = np.expand_dims(img_idxs, axis=1)
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1]*self.img_wh[2], self.batch_size)
            gray = self.imgs[img_idxs,pix_idxs]
            sample = {'img_idxs': img_idxs,'pix_idxs': pix_idxs,
                      'gray': gray[:,:1],
                      't':time_idxs}
        else:
            sample = {'img_idxs': idx, 't':np.repeat(np.expand_dims(np.array([idx]), axis=1),self.img_wh[0]*self.img_wh[1],axis=0)}
        return sample