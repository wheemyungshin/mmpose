import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .se_layer import SELayer

class GhostModule(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                ratio=2,
                dw_size=3,
                relu=True,
                with_cp=False):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        self.with_cp = with_cp
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
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

class GhostBottleneck(nn.Module):
    def __init__(self,
                in_channels,
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                se_cfg=None,
                with_cp=False,                    
                groups=None,#not used
                with_expand_conv=True,#not used
                conv_cfg=None,#not used
                norm_cfg=dict(type='BN'),#not used
                act_cfg=dict(type='ReLU')):#not used
        super(GhostBottleneck, self).__init__()
        
        self.with_se = se_cfg is not None
        if self.with_se:
            assert isinstance(se_cfg, dict)
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride,
                             padding=(kernel_size-1)//2,
                             groups=mid_channels, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_channels)

        # Squeeze-and-excitation
        if self.with_se:
            self.se = SELayer(**se_cfg)

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)
        
        # shortcut
        if (in_channels == out_channels and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                       padding=(kernel_size-1)//2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.with_se:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x