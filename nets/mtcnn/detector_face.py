# 人脸框检测器
import numpy as np
import torch
from torch.autograd import Variable
from nets.mtcnn.get_nets import PNet, RNet, ONet
from nets.mtcnn.first_stage import run_first_stage
from utils.box_utils import nms, calibrate_box, convert_to_square, get_image_boxes
from utils.img_utils import pyramid_image

class MtcnnModel(object):
    def __init__(self):
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet().eval()

    def detectFace(self, image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        # 构建图像金字塔
        scales = pyramid_image(image, min_face_size)
        """Stage1-PNet"""
        bounding_boxes = []

        # 不同的尺度运行pnet网络
        for s in scales:
            boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        if len(bounding_boxes) == 0:
            return [], []

        # 预测框校准
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        # 非极大抑制
        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # 使用offsets进行转换
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])

        # 转换为正方形
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        """Stage2-RNet"""
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        # img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        img_boxes = torch.FloatTensor(img_boxes)
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        # 校准
        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        """Stage3-ONet"""
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        # img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        img_boxes = torch.FloatTensor(img_boxes)

        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # 计算landmard点, 比例换算
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

def detect_faces(image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
    """人脸检测
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.
    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """
    # 模型载入
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval() #针对模型的测试；onet.train()针对模型的训练

    # 构建图像金字塔
    scales = pyramid_image(image, min_face_size)

    """Stage1-PNet"""
    bounding_boxes = []

    # 不同的尺度运行pnet网络
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # 预测框校准
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    # 非极大抑制
    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # 使用offsets进行转换
    bounding_boxes = calibrate_box(bounding_boxes[:,0:5], bounding_boxes[:, 5:])

    # 转换为正方形
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    """Stage2-RNet"""
    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    # img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    img_boxes = torch.FloatTensor(img_boxes)
    output = rnet(img_boxes)
    offsets = output[0].data.numpy() # shape [n_boxes, 4]
    probs = output[1].data.numpy() # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    # 校准
    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    """Stage3-ONet"""
    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if len(img_boxes) == 0:
        return [], []
    # img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
    img_boxes = torch.FloatTensor(img_boxes)

    output = onet(img_boxes)
    landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].data.numpy()  # shape [n_boxes, 4]
    probs = output[2].data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # 计算landmard点, 比例换算
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks


