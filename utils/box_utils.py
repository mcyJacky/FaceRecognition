import numpy as np
from PIL import Image
import torch

#################################################################
# MTCNN
#################################################################
def _preprocess(img):
    """Preprocessing step before feeding the network.
    Arguments:
        img: a float numpy array of shape [h, w, c].
    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5)*0.0078125 # 图片归一化 (-1,1)
    return img

def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.
    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.
    Returns:
        list with indices of the selected boxes
    """
    # if there is no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxex
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)

    while len(ids) > 0:
        # 取出最大置信度
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # 交叉框左上角
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # 交叉框右下角
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # 交叉框的宽度和高度
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # 交叉框面积
        inter = w * h
        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]]))

    return pick

def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5 # 计算时，先将坐标转到中心，然后计算
    square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes

def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].
    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    w = np.expand_dims(w, axis=1)  # (n,1)
    h = np.expand_dims(h, axis=1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    translation = np.hstack([w, h, w, h])
    translation = translation * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def get_image_boxes(bboxes, img, size=24):
    """Cut out boxes from the image(如将pnet获得的bbox取出).
    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.
    Returns:
        a float numpy array of shape [n, 3, size, size].
    """
    num_boxes = len(bboxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = _correct_bboxes(bboxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')
        img_array = np.asarray(img, 'uint8')

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, 'float32')

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes

def _correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
        with respect to cutouts.
    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.
    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.
        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] - (ex[ind] - width) - 2.0
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list

#################################################################
# Retinaface
#################################################################
def decode(loc, priors, variances):
    """# 对先验框进行解码调整，获得中心预测框
    :param loc: lacation predictions for loc layers. [37840, 4]
    :param priors: 先验框 [37840, 4]
    :param variances: 方差 [0.1, 0.2
    :return:boxes
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],  # 中心调整按公式
                       priors[:, 2:] * torch.exp(variances[1] * loc[:, 2:])), dim=1)  # 长宽调整按公式

    # 转换为左上和右下角
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

# 对先验框进行调整，获得人脸关键点
def decode_landm(pre, priors, variances):
    # @pre: [37840, 10]
    landms = torch.cat((priors[:, :2] + priors[:, 2:] * variances[0] * pre[:, :2],
                        priors[:, :2] + priors[:, 2:] * variances[0] * pre[:, 2:4],
                        priors[:, :2] + priors[:, 2:] * variances[0] * pre[:, 4:6],
                        priors[:, :2] + priors[:, 2:] * variances[0] * pre[:, 6:8],
                        priors[:, :2] + priors[:, 2:] * variances[0] * pre[:, 8:10],
                       ), dim=1)
    return landms

# 计算交并比
def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou

# 非极大抑制
def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.3):
    detection = boxes
    # 取阈值大于置信度的框
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]
    if not np.shape(detection)[0]:
        return []

    best_box = []
    # 对置信度从高至低排序
    scores = detection[:, 4]
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    # 去除重合度比较大的框
    while np.shape(detection)[0] > 0:
        best_box.append(detection[0])
        if len(detection) == 1:
            break

        ious = iou(best_box[-1], detection[1:])
        detection = detection[1:][ious < nms_thres]

    return np.array(best_box)

#################################################################
# FaceNet
#################################################################
# 矩形转换为正方形
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    max_side = np.maximum(w, h)
    rectangles[:,0] = rectangles[:,0] + w*0.5 - max_side*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - max_side*0.5
    rectangles[:,2:4] = rectangles[:,0:2] + max_side
    return rectangles

# 转换landmark坐标
def convert_landmark_coordinate(landmarks):
    # [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5] -> [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    new_landmarks = np.zeros_like(landmarks)
    new_landmarks[:, 0] = landmarks[:, 0]
    new_landmarks[:, 1] = landmarks[:, 5]
    new_landmarks[:, 2] = landmarks[:, 1]
    new_landmarks[:, 3] = landmarks[:, 6]
    new_landmarks[:, 4] = landmarks[:, 3]
    new_landmarks[:, 5] = landmarks[:, 7]
    new_landmarks[:, 6] = landmarks[:, 4]
    new_landmarks[:, 7] = landmarks[:, 8]
    new_landmarks[:, 8] = landmarks[:, 5]
    new_landmarks[:, 9] = landmarks[:, 9]

    return new_landmarks

# 计算人脸距离
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

# 人脸比较
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance)
