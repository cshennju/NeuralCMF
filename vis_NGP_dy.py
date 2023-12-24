import torch
import os
import numpy as np
from einops import rearrange
from kornia.utils import create_meshgrid3d
from models.networks_dy_siren import Siren,TimeEmbedding
import argparse
from utils import slim_ckpt, load_ckpt
import warnings; warnings.filterwarnings("ignore")
import scipy


def parse_args():
    parser = argparse.ArgumentParser()
     # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='heart_dy_3d',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv'],
                        help='which dataset to train/test')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_transient_flow = ['fw', 'bw']
    ne_fine = Siren().cuda()
    load_ckpt(ne_fine, f'./ckpts/{args.dataset_name}/{args.root_dir}/epoch=9_slim.ckpt')
    @torch.no_grad()
    def ne_func(points,t,flow,embedding_t_):
        t_embedded_ = embedding_t_(t)
        result = ne_fine(points,t_embedded_,flow)
        res = result[...,:1]
        #fw = result[...,1:4]
        #bw = result[...,4:7]
        return res
    
    embedding_t = TimeEmbedding(4).cuda()
    grid = create_meshgrid3d(130,110,140, normalized_coordinates=False)[0].cuda()
    print(grid.min())
    grid = grid/140
    rays_dir = grid.reshape(-1, 3)

    t = torch.ones((len(rays_dir),1)).cuda()
    ts = 0*t #change ts to get which frame you want
    res = ne_func(rays_dir,ts,output_transient_flow,embedding_t)
    res = rearrange(res.cpu().numpy(), '(h w d) c -> h w d c', h=130,w=110)
    res = res.squeeze(3)
    res = (res * 255).astype(np.uint8)
    filename = os.path.basename(args.root_dir)
    scipy.io.savemat(os.path.join('./straus3d/',filename +'_3d.mat'),{'rgb':res})