import os
import numpy as np
from einops import rearrange
import torch
from models.networks_dy_siren import Siren,TimeEmbedding
import argparse
from utils import slim_ckpt, load_ckpt
import warnings; warnings.filterwarnings("ignore")
from kornia.utils import create_meshgrid3d

def parse_args():
    parser = argparse.ArgumentParser()
     # dataset parameters
    parser.add_argument('--root_dir', type=str, default='./straus/normal_s_siren',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='heart_dy_3d',
                        choices=['heart', 'heart_dy', 'colmap', 'nerfpp', 'rtmv'],
                        help='which dataset to train/test')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ne_fine = Siren().cuda()
    load_ckpt(ne_fine, f'./ckpts/{args.dataset_name}/{args.root_dir}/epoch=9_slim.ckpt')
    output_transient_flow = ['fw','bw']

    @torch.no_grad()
    def ne_func(points,t,flow,embedding_t_):
        t_embedded_ = embedding_t_(t)
        result = ne_fine(points,t_embedded_,flow)
        fw = result[...,1:4]
        
        return fw 

    name = 'normal'
    grid = create_meshgrid3d(130,110,140, False)[0] 
    grid = grid/140
    i, j, k = grid.unbind(-1)
    directions = torch.stack([i,k,j], -1)
    directions = directions.reshape(-1, 3)

    img_out = os.path.join('straus_warp/'+name)
    embedding_t = TimeEmbedding(4).cuda()
    os.makedirs(img_out, exist_ok=True)
    t = torch.ones((len(directions),1)).float().cuda()
    rays_d = torch.tensor(directions).float().cuda()
    fwd = []
    f_w = 0
    for j in range(4):
        ts = j*t
        fw = ne_func(rays_d+f_w,ts,output_transient_flow,embedding_t)
        fw_new = fw.cpu().numpy()
        fw_new = rearrange(fw_new, '(h w d) c -> h w d c', h=130,w=110)
        print(fw_new.shape)
        f_w = f_w + fw
        fwd.append(fw_new)
    fwd = np.stack(fwd)
    np.save(os.path.join(img_out, 'flow_'+name+'.npy'),fwd)
