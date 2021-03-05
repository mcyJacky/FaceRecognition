# FaceRecognition
Face recognition with mtcnn/retinaface/facenet

### 主要安装库
- keras
- pytorch
- numpy
- pillow
- cv2 (opencv库)

### 运行环境
- python3.6
- pycharm
- anaconda(jupyter notebook)

### 人脸检测步骤
1.MTCNN算法进行人脸检测
```
# 对一张测试图片进行人脸检测如下（或使用pycharm运行）
python detector_mtcnn.py
```

2.Retinaface算法进行人脸检测
```
# 对一张测试图片进行人脸检测如下（或使用pycharm运行）
python detector_retinaface.py
```

### 人脸识别步骤
1.将识别图片添加至./face_dataset/datset文件夹中作为人脸库
2.运行FaceRecognize.py进行人脸视频识别
```
python FaceRecognize.py
```
