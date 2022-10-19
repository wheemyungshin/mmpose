# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class KLDiscretLossSimCC(nn.Module):
    def __init__(self, use_target_weight=True):
        super(KLDiscretLossSimCC, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1) #[B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')
        self.use_target_weight = use_target_weight
 
 
    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1) 
        return loss

    def forward(self, output, target, target_weight):
        output_x = output[0]
        output_y = output[1]
        target_x = target[0]
        target_y = target[1]
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:,idx].squeeze()
            coord_y_pred = output_y[:,idx].squeeze()
            coord_x_gt = target_x[:,idx].squeeze()
            coord_y_gt = target_y[:,idx].squeeze()            
            if self.use_target_weight:
                weight = target_weight[:,idx].squeeze()
                loss += (self.criterion(coord_x_pred,coord_x_gt).mul(weight).mean()) 
                loss += (self.criterion(coord_y_pred,coord_y_gt).mul(weight).mean())
            else:
                loss += (self.criterion(coord_x_pred,coord_x_gt).mean()) 
                loss += (self.criterion(coord_y_pred,coord_y_gt).mean())
        
        return loss / num_joints 
        
@LOSSES.register_module()
class JointsMSELossSimCC(nn.Module):
    """
    Not implemeted
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        output_x = output[0]
        output_y = output[1]
        target_x = target[0]
        target_y = target[1]
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred * target_weight[:, idx],
                                       heatmap_gt * target_weight[:, idx])
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight
