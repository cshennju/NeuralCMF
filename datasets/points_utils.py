import torch
import numpy as np
from kornia import create_meshgrid
from kornia.utils import create_meshgrid3d
from einops import rearrange


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ini_position_3d(D,H, W, device='cpu', flatten=True):
    
    grid = create_meshgrid3d(D, H, W, False, device=device)[0] # (H, W, 2)
    grid = grid/160
    i, j, k = grid.unbind(-1)

    ini_position = torch.stack([i,k,j], -1)

    if flatten:
        ini_position = ini_position.reshape(-1, 3)
        ini_position = ini_position.requires_grad_(True)
    return ini_position

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ini_position(H, W, device='cpu', flatten=True):
    
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    #print(grid)
    i, j = grid.unbind(-1)

    ini_position = \
            torch.stack([i,j,80*torch.ones_like(i)], -1)
    if flatten:
        ini_position = ini_position.reshape(-1, 3)
    return ini_position

@torch.cuda.amp.autocast(dtype=torch.float32)
def get_points(ini_position, c2w):
    
    points_r = rearrange(ini_position, 'n c -> n 1 c') @ \
                 rearrange(c2w[..., :3], 'n a b -> n b a')
    points_r = rearrange(points_r, 'n 1 c -> n c')
    points_t = c2w[..., 3].expand_as(points_r)
    points_d = points_r + points_t
    points_d = points_d/160-0.5
    return points_d


@torch.cuda.amp.autocast(dtype=torch.float32)
def axisangle_to_R(v):
    v_ndim = v.ndim
    if v_ndim==1:
        v = rearrange(v, 'c -> 1 c')
    zero = torch.zeros_like(v[:, :1]) # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v)/norm_v)*skew_v + \
        ((1-torch.cos(norm_v))/norm_v**2)*(skew_v@skew_v)
    if v_ndim==1:
        R = rearrange(R, '1 c d -> c d')
    return R

@torch.cuda.amp.autocast(dtype=torch.float32)
def R2axangle(R):
    theta = torch.acos((R.trace()-1)/2)
    if theta == 0:
        w = torch.tensor([0,0,0])
    else:
        aix1 = R[2,1]-R[1,2]
        aix2 = R[0,2]-R[2,0]
        aix3 = R[1,0]-R[0,1]
        aix = torch.tensor([aix1,aix2,aix3])
        w = theta * (1 / (2 * torch.sin(theta))) * aix

    return w