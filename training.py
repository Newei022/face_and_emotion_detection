import os
import pickle
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import optimizers
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten, Conv2D

# 训练模型

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 128
nb_validation_samples = 3589

# 生成器读取图像
train_dir = r'.\dataset\train'
val_dir = r'.\dataset\val'
test_dir = r'.\dataset\test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 重放缩因子，数值乘以1.0/255（归一化）
    shear_range=0.2,  # 剪切强度（逆时针方向的剪切变换角度）
    zoom_range=0.2,  # 随机缩放的幅度
    horizontal_flip=True  # 进行随机水平翻转
)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
)

# 构建网络
model = Sequential()
# 第一段
# 第一卷积层，64个大小为5×5的卷积核，步长1，激活函数relu，卷积模式same，输入张量的大小
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 第一池化层，池化核大小为2×2，步长2
model.add(BatchNormalization())  # 归一化是一种常用于数据预处理的方法。根据需求定义将数据约束到固定的一定范围。希望转化后的数值满足一定的特性（分布）。
model.add(Dropout(0.4))  # 随机丢弃40%的网络连接，防止过拟合
# 第二段
model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
# 第三段
model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())  # 过渡层
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))  # 全连接层
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))  # 分类输出层
model.summary()

# 编译
model.compile(loss='categorical_crossentropy',
              # optimizer=optimizers.Adam(),  # Adam优化器
              # optimizer=optimizers.RMSprop(learning_rate=0.0001),  # rmsprop优化器
              optimizer=optimizers.RMSprop(),  # rmsprop优化器
              metrics=['accuracy'])
start_time = time.time()  # 记录程序开始运行时间
# 训练模型
history = model.fit(
    train_generator,  # 生成训练集生成器
    # steps_per_epoch=None, #一个epoch包含的步数（每一步是一个batch的数据送入），当使用如TensorFlow数据Tensor之类的输入张量进行训练时，默认的None代表自动分割，即数据集样本数/batch样本数。
    steps_per_epoch=243,  # train_num/batch_size=128
    epochs=1000,  # 数据迭代轮数
    # validation_data=None, #这个参数会覆盖 validation_split，即两个函数只能存在一个，它的输入为元组 (x_val，y_val)，这作为验证数据。
    validation_data=validation_generator,  # 生成验证集生成器
    # validation_steps=None, #在验证集上的step总数，仅当steps_per_epoch被指定时有用。
    validation_steps=28  # valid_num/batch_size=128
)
end_time = time.time()  # 记录程序结束运行时间
########################

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()



# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.grid(True)
# plt.savefig('test_7-1_train_loss.png')



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
# fig.tight_layout()
plt.savefig('test_7-1_train_acc.png')


#########################

# 评估模型
test_loss, test_acc = model.evaluate(test_generator, steps=28)
print("test_loss: %.4f - test_acc: %.4f" % (test_loss, test_acc * 100))

###############

test_generator = test_datagen.flow_from_directory(
    test_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)

class_labels = test_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

Y_pred = model.predict_generator(test_generator, nb_validation_samples // batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)

emotions = ["愤怒", "厌恶", "害怕", "快乐", "悲伤", "惊讶", "自然"]
# target_names = list(class_labels.values())
print(classification_report(test_generator.classes, y_pred, target_names=emotions))

###############

# 保存模型
model_json = model.to_json()
with open('model/myModel_2_json.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('test_7-1_train__weight.h5')
model.save('train.h5')

with open('test_7-1_train_log.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt, 0)

with open('test_7-1_train_time.txt', 'w') as file_time:
    file_time.write(str(end_time - start_time))

##########################

parent = "model"
if not os.path.exists(parent):
    os.mkdir(parent)

model.save("%s/fer25.h5" % parent)

###########################



class_labels = ["angry","disgust","fear","happy","sad","surprise","natural"]
# 创建混淆矩阵
cm = confusion_matrix(test_generator.classes, y_pred, normalize='true')

# 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=class_labels, yticklabels=class_labels,
       ylabel='True label',
       xlabel='Predicted label')

# 在矩阵中添加文本标签
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], '.2f'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
# plt.show()
plt.savefig('test_7-1_train_hunxiao.png')
