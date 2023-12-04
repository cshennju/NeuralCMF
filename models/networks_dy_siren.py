import torch
from torch import nn
import numpy as np
import math
import torch.nn.parallel
import numpy as np

class TimeEmbedding(nn.Module):
    def __init__(self,T):
        super().__init__()
        self.T = T

    def forward(self, x):

        x = (x%self.T)/self.T

        return x

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features) / self.omega_0, 
                                             np.sqrt(6/self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
class Siren(nn.Module):
    def __init__(self, pos_in_dims=3,D=64,hidden_layers=4, outermost_linear=True, 
                 first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(pos_in_dims, D, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(D, D, 
                                      is_first=False, omega_0=hidden_omega_0))
        

        if outermost_linear:
            final_linear = nn.Linear(D, 32)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / D) / hidden_omega_0, 
                                              np.sqrt(6 / D) / hidden_omega_0)
            self.net.append(final_linear)
                
        else:
            self.net.append(SineLayer(D, 32, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

        self.net1 = []
        self.net1.append(SineLayer(pos_in_dims+1, D, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net1.append(SineLayer(D, D, 
                                      is_first=False, omega_0=hidden_omega_0))
        

        if outermost_linear:
            final_linear1 = nn.Linear(D, 32)
            
            with torch.no_grad():
                final_linear1.weight.uniform_(-np.sqrt(6 / D) / hidden_omega_0, 
                                              np.sqrt(6 / D) / hidden_omega_0)
            self.net1.append(final_linear1)
                
        else:
            self.net1.append(SineLayer(D, 32, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net1 = nn.Sequential(*self.net1)

        self.rgb_net = SineLayer(64, 64, is_first=False, omega_0=hidden_omega_0)

        self.transient_rgb = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.fw = nn.Sequential(nn.Linear(64, 3), nn.Tanh())
        self.bw = nn.Sequential(nn.Linear(64, 3), nn.Tanh())

    def forward(self, x,t,output_transient_flow=[]):
        h0 = self.net1(torch.cat([x,t], 1))
        h1 = self.net(x)
        h3 = self.rgb_net(torch.cat([h0,h1], 1))
        density = self.transient_rgb(h3)
        transient_list = [density]
        if 'fw' in output_transient_flow:
            transient_list += [self.fw(h3)]
        if 'bw' in output_transient_flow:
            transient_list += [self.bw(h3)]
        transient = torch.cat(transient_list, 1)

        return transient
