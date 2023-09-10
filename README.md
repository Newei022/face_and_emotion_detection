# face_and_emotion_detection

##基本介绍
Enhanced_data.py文件是用于数据增强，将图片进行翻转，缩放，反转等操作

opcvface2.py文件是运行摄像头实现实时的人脸表情识别

resort.py文件是开源数据集fer2013进行标签分类处理

开源数据集地址：https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Shear.py文件是扩充数据集用到的程序，可以将网络上获取的图片进行人脸检测和分割

training.py文件是训练模型

transfor.py解析从csv文件中解析成图片

人脸表情识别，训练集的准确率可以达到95%，测试集的准确率能达到68%。
