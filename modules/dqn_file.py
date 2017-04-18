"""
Depp Q network (DQN) class:

Refs:

[1] DQN paperby DeepMind

[2] An implementation of a simpler game in PyTorch at http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""

import math
import torch.nn as nn
import torch.nn.functional as F

# Helper function that compute the output of a cross correlation
def dim_out(dim_in,ks,stride):
    return math.floor((dim_in-ks)/stride+1)

class dqn(nn.Module):
        
    def __init__(self, C_in, C_H, C_out, ks, stride,
                 obs_window_H, obs_window_W):
        """hp = hyperparameters, dictionary"""
        super(dqn, self).__init__()
        # Conv2D has arguments C_in, C_out, ... where C_in is the number of input channels and C_out that of
        # output channels, not to be confused with the size of the image at input and output which is automatically
        # computed given the input and the kernel_size (ks). 
        # Further, in the help, (N,C,H,W) are resp. number of samples, number of channels, height, width.
        # Note: that instead nn.Linear requires both number of input and output neurons. The reason is that
        # conv2d only has parameters in the kernel, which is independent of the number of neurons.
        # Note: we do not use any normalization layer
        self.C_H = C_H
        self.conv1 = nn.Conv2d(C_in, self.C_H, kernel_size=ks, stride=stride)
        self.H1 = dim_out(obs_window_H,ks,stride)
        self.W1 = dim_out(obs_window_W,ks,stride)
        in_size = self.C_H*self.W1*self.H1
        self.lin1 = nn.Linear(in_size, in_size) #lots of parameters!
        self.conv2 = nn.Conv2d(self.C_H, self.C_H, kernel_size=ks, stride=stride)
        H2 = dim_out(self.H1,ks,stride)
        W2 = dim_out(self.W1,ks,stride)
        in_size = self.C_H*W2*H2
        self.lin2 = nn.Linear(in_size, C_out)

    def forward(self, x):
        # Apply rectified unit (relu) after each layer
        x = F.relu(self.conv1(x))
        # to feed into self.lin. we reshape x has a (size(0), rest) tensor where size(0) is number samples.
        # -1 tells it to infer size automatically.
        x = x.view(x.size(0), -1) 
        x = F.relu(self.lin1(x))
        # reshape to feed it into conv2, this time:
        x = x.view(x.size(0), self.C_H, self.H1, self.W1) 
        x = F.relu(self.conv2(x))
        # reshape to feed it into lin2, this time:
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin2(x))    
        return x
