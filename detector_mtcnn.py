from nets.mtcnn.detector_face import detect_faces, MtcnnModel
from utils.visualization_utils import show_bboxes
from PIL import Image
import torch

if __name__ == '__main__':
    print(torch.__version__) #1.4.0
    # print(torch.FloatTensor([[1,2,3]], requires_grad=True))
    # x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)
    # print(x)
    mtcnnModel = MtcnnModel()

    print("Predict start>>>>>")
    img = Image.open('face_dataset/test.jpg')
    print(img.size) #(w,h)
    # bounding_boxes, landmarks = detect_faces(img)
    bounding_boxes, landmarks = mtcnnModel.detectFace(img)
    print("bounding_box.shape:", bounding_boxes.shape, landmarks.shape)
    r_image = show_bboxes(img, bounding_boxes, landmarks)
    r_image.show()