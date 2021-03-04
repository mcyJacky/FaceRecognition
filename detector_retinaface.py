import cv2
from nets.retinaface.retinafaceEx import RetinafaceEx

if __name__ == '__main__':
    retinaface = RetinafaceEx()

    imgPath = 'face_dataset/office4.jpg'
    image = cv2.imread(imgPath)
    print("image.shape>>>>:", image.shape) # (h,w,3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r_image = retinaface.detectFace(image)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("result", r_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
