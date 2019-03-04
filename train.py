from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D,Dropout
from keras.optimizers import SGD
import cv2
import os
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
from load_dataset import load_data
import random


class DateSet:
    def __init__(self,dirPath,classnum):
        self.train_images=None
        self.train_labels=None
        self.test_images=None
        self.test_labels=None
        self.valid_images=None
        self.valid_labels=None
        self.input_shape=None
        self.dirPath=dirPath
        self.classnum=classnum
    def load(self):
        images,labels=load_data(self.dirPath)
        self.input_shape=images.shape[1:4]
        np.random.seed(10)
        self.train_images=np.random.permutation(images)
        np.random.seed(10)
        self.train_labels = np.random.permutation(labels)

        self.test_images=self.train_images[:100]
        self.test_labels = self.train_labels[:100]
        np.random.seed(10)
        self.test_images = np.random.permutation(self.test_images)
        np.random.seed(10)
        self.test_labels = np.random.permutation(self.test_labels)

        self.valid_images= self.train_images[100:200]
        self.valid_labels = self.train_labels[100:200]
        np.random.seed(10)
        self.valid_images = np.random.permutation(self.valid_images)
        np.random.seed(10)
        self.valid_labels = np.random.permutation(self.valid_labels)

        self.train_images = self.train_images / 255
        self.test_images = self.test_images / 255
        self.valid_images = self.valid_images/255
        self.train_labels = np_utils.to_categorical(self.train_labels, self.classnum)
        self.valid_labels = np_utils.to_categorical(self.valid_labels, self.classnum)
        self.test_labels = np_utils.to_categorical(self.test_labels, self.classnum)
'''

        i1=random.randint(0,300)
        i2 = random.randint(0, 100)
        i3 = random.randint(0, 100)
        print('train',self.train_labels[i1])
        print('test',self.train_labels[i2])
        print('valid',self.train_labels[i3])
        cv2.imshow('train',self.train_images[i1])
        cv2.imshow('test', self.train_images[i2])
        cv2.imshow('valid', self.train_images[i3])
        cv2.waitKey(0)
'''
class Model:
    def __init__(self):
        self.model=None
    def bulid_model(self,dataset,class_num=2):
        self.model=Sequential()

        self.model.add(Convolution2D(32, 3 ,3, border_mode='same',
                                     input_shape=dataset.input_shape,activation='relu'))

        self.model.add(MaxPooling2D(pool_size=2, strides=2,
                                    padding='same'))

        self.model.add(Convolution2D(64, 3 ,3, border_mode='same',
                                     activation='relu'))

        self.model.add(MaxPooling2D(pool_size=2, strides=2,
                                    padding='same'))
        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(Dense(1024))  # 14 Dense层,又被称作全连接层

        self.model.add(Dropout(0.5))
        self.model.add(Dense(512))  # 14 Dense层,又被称作全连接层

        self.model.add(Dropout(0.5))
        self.model.add(Dense(dataset.classnum))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果

        # 输出模型概况
        self.model.summary()
    def train(self, dataset, batch_size=100, nb_epoch=5):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        self.model.fit(dataset.train_images,
                       dataset.train_labels,
                       batch_size=batch_size,
                       nb_epoch=nb_epoch,
                       )
        loss, accuracy =  self.model.evaluate(dataset.test_images, dataset.test_labels)
        print('\ntest loss: ', loss)
        print('\ntest accuracy: ', accuracy)

    MODEL_PATH = 'model_3.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_models(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def num_predict(self, image):
        image = np.reshape(image, (1, 128, 128, 3))
        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        results = self.model.predict_proba(image)
        # 给出类别预测：0或者1
        result = self.model.predict_classes(image)
        # 返回类别预测结果
        print(result[0])
        print('识别系数', results[0])
        return result[0]

if __name__ == '__main__':
    '''
    dataSet=DateSet(r'D:\python\MyTest',2)
    dataSet.load()
    model=Model()
    model.bulid_model(dataSet)
    model.train(dataSet)
    model.save_model()
    '''
    model=Model()
    model.load_models('model_3.h5')
    for file in os.listdir('face/other'):
        image=cv2.imread('face/other/'+file)
        r=model.num_predict(image)
        if r == 0:
            print("zhang")
        if r == 1:
            print("sun")
