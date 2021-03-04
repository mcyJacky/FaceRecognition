import numpy as np
import math
import cv2

def pre_process(x):
    """图像标准化处理
    Arguments
        x: image
    Return: image standard result.
    """
    if x.ndim == 4:
        axis = (1,2,3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0,1,2)
        size= x.size
    else:
        raise ValueError('Image dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean)/ std_adj
    return y

def l2_norm(x, axis=-1, epsilon=1e-10):
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

def pyramid_image(image, min_face_size = 15.0):
    """图像金字塔
        Arguments
            image: an instance of PIL.Image
            min_face_size: a float number
        Return: image pyramid scales.
    """
    # 最小检测尺寸
    min_detection_size = 12
    factor = 0.707

    # 最大缩放比例
    max_scale = min_detection_size / min_face_size

    width, height = image.size
    min_length = min(width, height)
    min_length *= max_scale

    # 缩放比率列表
    scales = []
    factor_count = 0
    while min_length > min_detection_size:
        scales.append(max_scale * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    print('scales:', ['{:.2f}'.format(s) for s in scales])
    print('number of different scales:', len(scales))
    return scales

def Alignment_1(img, landmark):
    '''人脸根据两眼角度对齐
    :param img: (h, w, c)
    :param landmark: (5, 2)
    :return: new_img, new_landmark
    '''
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]

    if x == 0:
        angle = 0
    else:
        angle = math.atan(y/x) * 180 / math.pi

    center = (img.shape[1]//2, img.shape[0]//2)

    # 旋转矩阵
    rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img, rotationMatrix, (img.shape[1], img.shape[0]))

    # 关键点转换
    rotationMatrix = np.array(rotationMatrix)
    # print("RotationMatrix.shape>>>", rotationMatrix.shape) #(2, 3)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(rotationMatrix[0, 0] * landmark[i, 0] + rotationMatrix[0, 1] * landmark[i, 1] + rotationMatrix[0, 2])
        pts.append(rotationMatrix[1, 0] * landmark[i, 0] + rotationMatrix[1, 1] * landmark[i, 1] + rotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark