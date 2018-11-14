from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .attention import PAM_Module, CAM_Module

__all__ = ['DenseNet121PAM']


class DenseNet121PAM(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(DenseNet121PAM, self).__init__()
        self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.danet_head = DANetHead(1024, 1024, nn.BatchNorm2d)
        self.classifier = nn.Linear(256 + 1024, num_classes)

    def forward(self, x):
        x = self.base(x)
        base_x = x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        pa, pose, pose_mask = self.danet_head(base_x)
        pa = F.avg_pool2d(pa, pa.size()[2:])
        pa = pa.view(pa.size(0), -1)
        f = torch.cat((x, pa), 1)
        if not self.training:
            pose_mask = F.max_pool2d(pose_mask, 1)
            return f, pose, pose_mask
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y, pose
        elif self.loss == {'xent', 'htri'}:
            return y, f, pose
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        # inter_channels = in_channels
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.pa = PAM_Module(inter_channels)

        self.pose_reg_fc = nn.Sequential(nn.Dropout2d(0.1, False),
                                         nn.Linear(inter_channels, 18 * 2)) # 18 points, 36 coordinates

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        pa_feat, pa_mask = self.pa(feat1)
        pa_conv = self.conv51(pa_feat)
        pa_output = self.conv6(pa_conv)

        pooled_pa_mask = F.avg_pool2d(pa_mask, pa_mask.size()[2:])
        pooled_pa_mask = pooled_pa_mask.view(pooled_pa_mask.size(0), -1)
        pose_output = self.pose_reg_fc(pooled_pa_mask)

        output = [pa_output, pose_output, pa_mask]
        return tuple(output)
