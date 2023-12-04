import torch

@torch.cuda.amp.autocast()
def render(model, xyz, embedding_t,output_transient_flow,ts,T):

    results = {}
    typ = 'fine'

    results[f'xyzs_{typ}'] = xyz
    t_embedded_ = embedding_t(ts)

    out = model(xyz,t_embedded_,output_transient_flow=output_transient_flow)
    
    results[f'transient_grays_{typ}'] = transient_grays = out[...,:1]
    results['transient_flows_fw'] = transient_flows_fw = out[..., 1:4]
    results['transient_flows_bw'] = transient_flows_bw = out[..., 4:7]

    results['xyzs_fw'] = xyz_fw = xyz + transient_flows_fw
    xyz_fw_ = xyz_fw
    tp1_embedded_ = embedding_t(ts+1) 
    results['gray_fw'], transient_flows_fw_bw = \
        render_transient_warping(model,xyz_fw_, tp1_embedded_, 'bw')

    results['xyzs_bw'] = xyz_bw = xyz + transient_flows_bw
    xyz_bw_ = xyz_bw
    tm1_embedded_ = embedding_t(ts-1) 
    results['gray_bw'], transient_flows_bw_fw = \
        render_transient_warping(model,xyz_bw_, tm1_embedded_, 'fw')
    results['xyzs_fw_bw'] = xyz_fw + transient_flows_fw_bw
    results['xyzs_bw_fw'] = xyz_bw + transient_flows_bw_fw
    
    transient_flows_fw_T = 0
    for i in range(T):
        i = i+1
        t_embedded_T = embedding_t(ts+i) # t+1
        xyz_fw = xyz_fw + transient_flows_fw_T
        result, transient_flows_fw_T = \
            render_transient_warping(model,xyz_fw, t_embedded_T, 'fw')          
    results['gray_T'] = result
    results['xyzs_T'] = xyz_fw

    transient_flows_bw_T = 0
    for i in range(T):
        i = i+1
        t_embedded_bT = embedding_t(ts-i) # t+1
        xyz_bw = xyz_bw + transient_flows_bw_T
        resultb, transient_flows_bw_T = \
            render_transient_warping(model,xyz_bw, t_embedded_bT, 'bw')         
    results['gray_bT'] = resultb
    results['xyzs_bT'] = xyz_bw
    
    results[f'gray_{typ}'] = transient_grays

    return results

@torch.cuda.amp.autocast()
def render_transient_warping(model,xyz, t_embedded, flow):
           
    out = model(xyz,t_embedded,output_transient_flow=[flow])
    transient_grays_w = out[..., :1]
    transient_flows_w = out[..., 1:4]
    
    return transient_grays_w, transient_flows_w