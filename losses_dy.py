import torch
from torch import nn
from einops import reduce

class NeuralCMFLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,inputs, targets):
        ret = {}
        ret['col_l'] = reduce((inputs['gray_fine']-targets['gray'])**2,
                              'n1 c -> n1', 'mean')
        pho_w = 1.0
        ret['pho_l'] = pho_w * (inputs['gray_fw']-targets['gray'])**2
        ret['pho_l']+= pho_w * (inputs['gray_bw']-targets['gray'])**2
        ret['pho_l'] = reduce(ret['pho_l'], 'n1 c -> n1', 'mean')

        pho_w2 = 1.0
        ret['pho_l2'] = pho_w2 * (inputs['gray_T']-targets['gray'])**2
        ret['pho_l2']+= pho_w2 *(inputs['gray_bT']-targets['gray'])**2
        ret['pho_l2'] = reduce(ret['pho_l2'], 'n1 c -> n1', 'mean')
        
        cyc_w = 0.1
        ret['cyc_l'] = cyc_w * torch.abs(inputs['xyzs_fw_bw']-inputs['xyzs_fine'])
        ret['cyc_l']+= cyc_w * torch.abs(inputs['xyzs_bw_fw']-inputs['xyzs_fine'])
        ret['cyc_l'] = reduce(ret['cyc_l'], 'n1 c -> n1', 'mean')

        cyc_w2 = 0.1
        ret['cyc_l2'] = cyc_w2 * torch.abs(inputs['xyzs_T']-inputs['xyzs_fine'])
        ret['cyc_l2']+= cyc_w2 * torch.abs(inputs['xyzs_bT']-inputs['xyzs_fine'])
        ret['cyc_l2'] = reduce(ret['cyc_l2'], 'n1 c -> n1', 'mean')

        lambda_reg = 0.01
        xyzs_w = inputs['xyzs_fine']
        xyzs_fw_w = inputs['xyzs_fw']
        xyzs_bw_w = inputs['xyzs_bw']
        ret['reg_temp_sm_l'] = lambda_reg * torch.abs(xyzs_fw_w+xyzs_bw_w-2*xyzs_w)
        ret['reg_temp_sm_l'] = reduce(ret['reg_temp_sm_l'], 'n1 c -> n1', 'mean')
        ret['reg_min_l'] = lambda_reg * (torch.abs(xyzs_fw_w-xyzs_w)+
                                             torch.abs(xyzs_bw_w-xyzs_w))
        ret['reg_min_l'] = reduce(ret['reg_min_l'], 'n1 c -> n1', 'mean')
        
        for k, loss in ret.items():
            ret[k] = loss.mean()

        return ret

loss_dict = {'NeuralCMF': NeuralCMFLoss}