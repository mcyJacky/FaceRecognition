import cv2
import os
import numpy as np
from PIL import Image
from nets.facenet.inceptionResNet import InceptionResNetV1
from nets.mtcnn.detector_face import MtcnnModel
from utils.visualization_utils import show_bboxes
from utils.box_utils import rect2square, convert_landmark_coordinate, compare_faces, face_distance
from utils.img_utils import Alignment_1

# 基于MTCNN网络进行人脸检测-人脸识别
class FaceRecognize_MTCNN(object):
    _defaults = {
        'originalPath': './face_dataset/dataset'
    }

    def __init__(self):
        self.__dict__.update(self._defaults)

        # 人脸检测网络
        self.detector_model = MtcnnModel()
        # 载入facenet网络
        self.facenet_model = InceptionResNetV1()

    def save_dataset_code(self):
        """
        对数据库中的人脸进行编码
        :known_face_encodings中存储的是编码后的人脸
        :nown_face_names为人脸的名字
        """
        self.known_face_encodings = []
        self.known_face_names = []

        face_list = os.listdir(self.originalPath)
        for face in face_list:
            name = face.split('.')[0]

            # read image
            img = cv2.imread(self.originalPath + '//' + face)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # detect face
            bounding_boxes, landmarks = self.detector_model.detectFace(Image.fromarray(img))

            # convert to square
            # bounding_boxes = rect2square(bounding_boxes)

            # new relative landmarks posistion
            landmarks = convert_landmark_coordinate(landmarks)

            # convert landmark to face image
            bounding_boxes = bounding_boxes[0]
            landmarks = landmarks[0]
            landmarks = (np.reshape(landmarks, (5,2)) - np.array([int(bounding_boxes[0]), int(bounding_boxes[1])])) / (bounding_boxes[3] - bounding_boxes[1]) * 160
            # print("conver_bbox>>>", bounding_boxes, '\n')
            # print("convert_landmarks_init>>>", np.reshape(landmarks, (5, 2)), '\n')
            # print("convert_landmarks>>>", landmarks)  # (1, 10)

            # crop image
            crop_img = img[int(bounding_boxes[1]):int(bounding_boxes[3]), int(bounding_boxes[0]):int(bounding_boxes[2])]
            crop_img = cv2.resize(crop_img, (160, 160))
            # cv2.imwrite(self.originalPath + '/crop/' + face, crop_img)

            # alignment
            new_img, _ = Alignment_1(crop_img, landmarks)
            # cv2.imwrite(self.originalPath + '/align/' + face, new_img)

            # encode with facenet
            new_img = np.expand_dims(new_img, 0)
            face_encoding = self.facenet_model.encode(new_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

    def recognize(self, image):
        '''
        :param image: (h, w, c)
        :return: image
        '''
        height, width, _ = np.shape(image)
        draw_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect face
        bounding_boxes, landmarks = self.detector_model.detectFace(Image.fromarray(draw_rgb))

        if len(bounding_boxes) == 0:
            return

        # convert to square(此mtcnn模型已校准)
        # bounding_boxes = rect2square(bounding_boxes)
        # bounding_boxes[:, 0] = np.clip(bounding_boxes[:, 0], 0, width)
        # bounding_boxes[:, 1] = np.clip(bounding_boxes[:, 1], 0, height)
        # bounding_boxes[:, 2] = np.clip(bounding_boxes[:, 2], 0, width)
        # bounding_boxes[:, 3] = np.clip(bounding_boxes[:, 3], 0, height)

        # new relative landmarks posistion
        landmarks = convert_landmark_coordinate(landmarks)

        # encoding
        face_encodings = []
        for bbox, landmark in zip(bounding_boxes, landmarks):
            # convert landmark to face image
            landmark = (np.reshape(landmark, (5, 2)) - np.array([int(bbox[0]), int(bbox[1])])) / (bbox[3] - bbox[1]) * 160

            # crop image
            crop_img = draw_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            crop_img = cv2.resize(crop_img, (160, 160))

            # align
            new_img, _ = Alignment_1(crop_img, landmark)

            # encode with facenet
            new_img = np.expand_dims(new_img, 0)
            face_encoding = self.facenet_model.encode(new_img)

            face_encodings.append(face_encoding)

        # compare coding distance
        face_names = []
        for face_encoding in face_encodings:
            # compare face code with dataset
            matches = compare_faces(self.known_face_encodings, face_encoding, tolerance=0.9)
            name = 'Unknown'
            # find min distance face
            face_distances = face_distance(self.known_face_encodings, face_encoding)
            best_index = np.argmin(face_distances)
            if matches[best_index]:
                name = self.known_face_names[best_index]
            face_names.append(name)

        # draw
        for bbox, name in zip(bounding_boxes, face_names):
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, name, (int(bbox[0]), int(bbox[3]) - 15), font, 0.75, (255, 255, 255), 2)

        return image

if __name__ == '__main__':
    faceDetetor = FaceRecognize_MTCNN()
    faceDetetor.save_dataset_code()

    # 人脸识别
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        faceDetetor.recognize(frame)
        cv2.imshow("video", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): #(0x71 == 0x71)
            break
    video_capture.release()
    cv2.destroyAllWindows()

