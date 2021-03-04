from math import ceil
import numpy as np
from itertools import product as product
import torch

class Anchors(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        """获取先验框
        :param cfg: config data
        :param image_size: (height, width)
        :param phase:
        """
        super(Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes'] # 先验框的基础边长[[16, 32], [64, 128], [256, 512]]
        self.steps = cfg['steps'] # 长和宽压缩的倍数 [8, 16, 32]
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        print('Anchors.feature_map:', self.feature_maps) #[[90, 160], [45, 80], [23, 40]]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 每个网格点2个先验框，都是正方形
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
