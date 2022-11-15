import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class GhostModule(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                ratio=2,
                dw_size=3,
                with_cp=False):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        self.with_cp = with_cp
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        def _inner_forward(x):
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.out_channels,:,:]
        
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out