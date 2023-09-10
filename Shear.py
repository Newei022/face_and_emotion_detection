import cv2
import os
import glob

# 剪裁的图片大小
size_m = 48
size_n = 48

# 人脸检测
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


cascade = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_alt2.xml')
for j in range(7):
    imglist = glob.glob('./extraimgs_befor/'+str(j)+'/*')
    i = 1
    for list in imglist:
        # print(list)
        # cv2读取图像
        img = cv2.imread(list)
        dst = img
        rects = detect(dst, cascade)
        for x1, y1, x2, y2 in rects:
            # 调整人脸截取的大小。横向为x,纵向为y
            roi = dst[y1 + 10:y2 + 20, x1 + 10:x2]
            img_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            re_roi = cv2.resize(img_roi, (size_m, size_n))
            # 新的图像存到data/image/jaffe_1
            f = "{}/{}".format("extraimgs_after", j)
            # print(f)
            if not os.path.exists(f):
                os.mkdir(f)
            # 切割图像路径
            # path = list.split(".")
            # name_info = list.split("\\")
            # suffix = name_info[1]
            # 新的图像存到data/image/jaffe_1   并以jpg 为后缀名
            cv2.imwrite('./extraimgs_after/{}/add_{}.jpg'.format(j, i), re_roi)
            print('add_{}.jpg'.format(i))
            i += 1