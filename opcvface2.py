import cv2
import numpy as np
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# 摄像头应用测试
face_classifier = cv2.CascadeClassifier("./dataset/haarcascade_frontalface_alt2.xml")
classifier = load_model("./model.h5")
class_labels = ["angry","disgust","fear","happy","sad","surprise","natural"]

def face_detector(img):
    # 将图像转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((48, 48), np.uint8), img
    return (x, w, y, h), roi_gray, img


cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 对ROI进行预测，然后查找类
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rect[0] + int((rect[1] / 2)), rect[2] + 25)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('All', image)
    if cv2.waitKey(1) == 27:  # 27是ESC键
        break

cap.release()
cv2.destroyAllWindows()