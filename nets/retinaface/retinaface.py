from nets.retinaface.mobilenet025 import MobileNetV1
from nets.retinaface.layers import FPN, SSH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.models._utils as _uils

# 预测先验框是否包含人脸num_anchor x 2
class ClassHead(nn.Module):
    def __init__(self, inchannels = 512, num_anchors = 2):
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 2, 1, stride = 1, padding = 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        print("ClassHead.out.shape:", out.shape)  # torch.Size([1, 90, 160, 4])
        return out.view(out.shape[0], -1, 2)

# 预测先验框Box
class BboxHead(nn.Module):
    def __init__(self, inchannels = 512, num_anchors = 2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, 1, stride = 1, padding = 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        print("BboxHead.out.shape:", out.shape)  # torch.Size([1, 90, 160, 8])
        return out.view(out.shape[0], -1, 4)

# 预测人脸关键点
class LandmarkHead(nn.Module):
    def __init__(self, inchannels = 512, num_anchors = 2):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, 1, stride = 1, padding = 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        print("LandmardHead.out.shape:", out.shape)  # torch.Size([1, 90, 160, 20])
        return out.view(out.shape[0], -1, 10)

# Retinaface网络
class Retinaface(nn.Module):
    def __init__(self, cfg = None, pretrained = False, phase = 'train'):
        """
        :param cfg: Network settings
        :param pretrained: isPretrained
        :param phase: train or eval
        """
        super(Retinaface, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pretrained:
                # TODO
                pass
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained = pretrained)

        self.body = _uils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        in_channels_stage2 = cfg['in_channel'] # 32
        in_channels_list = [
            in_channels_stage2 * 2, # 64
            in_channels_stage2 * 4, # 128
            in_channels_stage2 * 8  # 256
        ]
        out_channles = cfg['out_channel'] # 64

        self.fpn = FPN(in_channels_list, out_channles)
        self.ssh1 = SSH(out_channles, out_channles)
        self.ssh2 = SSH(out_channles, out_channles)
        self.ssh3 = SSH(out_channles, out_channles)

        self.ClassHead = self._make_class_head(fpn_num = 3, inchannels = cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num = 3, inchannels = cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num = 3, inchannels = cfg['out_channel'])

    def _make_class_head(self, fpn_num = 3, inchannels = 64, anchor_num = 2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num = 3, inchannels = 64, anchor_num = 2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num = 3, inchannels = 64, anchor_num = 2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        # ClassHead, BboxHead, LandmarkHead
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim = 1)
        print("classifications.shape:", classifications.shape, '\n')
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim = 1)
        print("bbox_regressions.shape:", bbox_regressions.shape, '\n')
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim = 1)
        print("ldm_regressions.shape:", ldm_regressions.shape, '\n')

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output