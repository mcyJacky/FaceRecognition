# retinaface扩展
import numpy as np
import torch
import torch.nn as nn
from nets.retinaface.config import cfg_mnet, cfg_re50
from nets.retinaface.retinaface import Retinaface
import os
from utils.anchors import Anchors
from utils.box_utils import decode, decode_landm, non_max_suppression
import cv2

def preprocess_input(image):
    # 每个通道减去均值，对图片进行白化
    image -= np.array((104, 117, 123), np.float32)
    return image

class RetinafaceEx(object):
    _defaults = {
        "model_path": 'model_data/retinaface/Retinaface_mobilenet0.25.pth',
        "confidence": 0.5,
        "backbone": 'mobilenet',
        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        self.__dict__.update(self._defaults)
        if self.backbone == 'mobilenet':
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._generate()

    def _generate(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' #指定cuda为0
        self.net = Retinaface(self.cfg, phase = 'eval').eval()
        # print("self.net:", self.net)
        print('Start Loading weights into state dict...')
        state_dict = torch.load(self.model_path, map_location = self.device)
        # print("state_dict", state_dict)
        self.net.load_state_dict(state_dict)
        print("Load weights finish!")

    def detectFace(self, image):
        """图片检测
        :param image: H x W x 3
        :return: image
        """
        # 图片类型转换
        old_image = image.copy()
        image = np.array(image, np.float32)
        print("image.shape:", image.shape) # (719, 1280, 3)
        im_height, im_width, _ = np.shape(image)

        # 将归一化后的框坐标转换成原图的大小
        scale = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]])
        print("scale:", scale) # tensor([1280.,  719., 1280.,  719.])
        scale_for_landmarks = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]])
        print("scale_for_landmarks:", scale_for_landmarks)

        # 预处理>pytorch格式
        image = preprocess_input(image).transpose(2, 0, 1) # (3，719, 1280)
        image = torch.from_numpy(image).unsqueeze(0)
        print("batch.image:", image.shape) # (1, 3, 719, 1280)

        # 计算先验框
        anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

        # 人脸检测
        with torch.no_grad():
            if self.cuda:
                scale = scale.cuda()
                scale_for_landmarks = scale_for_landmarks.cuda()
                image = image.cuda()
                anchors = anchors.cuda()

            loc, conf, landms = self.net(image)
            print("loc.shape:", loc.shape) # bbox torch.Size([1, 37840, 4])
            print("conf.shape:", conf.shape) # classification torch.Size([1, 37840, 2])
            print("landms.shape:", landms.shape, "\n") # landmark torch.Size([1, 37840, 10])

            # 预测框解码
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            print("boxes:", boxes.shape)  # (37840, 4)

            # 置信度
            conf = conf.data.squeeze(0)[:, 1:2].cpu().numpy()
            print("conf.shape:", conf.shape)  # (37840, 1)

            # 关键点解码
            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            landms = landms * scale_for_landmarks
            landms = landms.cpu().numpy()
            print("landms.shape:", landms.shape)  # (37840, 10)

            # 非极大抑制
            boxes_conf_landms = np.concatenate([boxes, conf, landms], -1)
            print("boxes_conf_landms1.shape:", boxes_conf_landms.shape)  # (37840, 15)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            print("boxes_conf_landms2.shape:", boxes_conf_landms.shape) #(49, 15)

        for b in boxes_conf_landms:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # 绘制人脸框
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # 绘制关键点
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

        return old_image