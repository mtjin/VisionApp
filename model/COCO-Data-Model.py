import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
class DataGenerator(Sequence):
    def __init__(self, p,b,o, y, batch_size, dim, dimy, n_channels, shuffle=False):
        self.X = p
        self.o = o
        self.b = b
        self.y = y if y is not None else y
        self.batch_size = batch_size
        self.dim = dim
        self.dimy = dimy
        self.n_channels = n_channels

        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __data_generation(self, X_list,b_list,o_list, y_list):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size,*self.dimy))
        temp = np.empty((self.batch_size,288,480))
        if y is not None:
            # 지금 같은 경우는 MNIST를 로드해서 사용하기 때문에
            # 배열에 그냥 넣어주면 되는 식이지만,
            # custom image data를 사용하는 경우
            # 이 부분에서 이미지를 batch_size만큼 불러오게 하면 됩니다.
            for index, (pointpath,bpath,opath, labelpath) in enumerate(zip(X_list,b_list,o_list, y_list)):
                # print(index)
                # print(pointpath)
                # print(bpath)
                # print(opath)
                # print(labelpath)
                mask = cv2.imread(labelpath, cv2.IMREAD_GRAYSCALE)
                dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
                img = cv2.imread(opath, cv2.IMREAD_COLOR)
                dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
                Point = cv2.imread(pointpath, cv2.IMREAD_GRAYSCALE)
                dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
                Back = cv2.imread(bpath, cv2.IMREAD_GRAYSCALE)
                dstBack = cv2.resize(Back, dsize=(480, 288), interpolation=cv2.INTER_AREA)
                tmask = np.array(dstmask)
                timg = np.array(dstimg)
                train_Point = np.array(dstPoint)
                train_Back = np.array(dstBack)
                X[index, :, :, 0:3] = timg
                X[index, :, :, 3] = train_Point
                X[index, :, :, 4] = train_Back
                X = X.reshape(len(X), 288, 480, 5).astype('float32') / 255.0
                temp[index, :, :] = tmask
                y = temp.reshape(len(temp), 288, 480, 1).astype('float32') / 255.0
            return X, y


    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X_list = [self.X[k] for k in indexes]
        b_list = [self.b[k] for k in indexes]
        o_list = [self.o[k] for k in indexes]
        if self.y is not None:
            y_list = [self.y[k] for k in indexes]
            X, y = self.__data_generation(X_list,b_list,o_list, y_list)
            return X, y
        else:
            y_list = None
            X = self.__data_generation(X_list,b_list,o_list, y_list)
            return X
def UNet():
    input_size = (288, 480, 5)
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inp, outputs=[conv10])

    return model



if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    pointdir = glob.glob('/home/su_j615/out_focusing/trainpoints10/*')
    backfiles = glob.glob('/home/su_j615/out_focusing/trainbackpoints10/*')
    maskfiles = glob.glob('/home/su_j615/out_focusing/trainmasks/*')
    valpointdir = glob.glob('/home/su_j615/out_focusing/valpoints10/*')
    length = int(len(pointdir))
    print(length)

    # Parameters
    params = {'dim': (288,480, 5),
              'dimy': (288,480, 1),
              'batch_size': 16,
              'n_channels': 1,
              'shuffle': False}

    # Train
    trainMask = []
    trainImg = []
    trainPoint = []
    trainBack = []

    # valid
    validMask = []
    validImg = []
    validPoint = []
    validBack = []

    for catindex in range(79):

        print(pointdir[catindex])
        catpointdir = glob.glob(pointdir[catindex]+'/*')
        catpointdir = sorted(catpointdir)
        for j in range(int(len(catpointdir))):
            pointfiles = glob.glob(catpointdir[j]+'/*')
            l = int(len(pointfiles))
            random.shuffle(pointfiles)
            mid = int(l)
            end = l
            for i in range(mid):

                base = os.path.basename(pointfiles[i])
                filename=os.path.splitext(base)[0]
                Imgpath = '/home/su_j615/out_focusing/train2017/' + filename[0:12] +'.jpg'
                maskpath = pointfiles[i].replace('/trainpoints10', '/trainmasks')
                pointindex = maskpath.split('/')[6]
                maskpath = maskpath.replace('/' + pointindex, '')
                backpath = pointfiles[i].replace('/trainpoints10', '/trainbackpoints10')
                if (not os.path.isfile(Imgpath) or not os.path.isfile(maskpath) or not os.path.isfile(backpath)):
                    continue
                trainPoint.append(pointfiles[i])
                trainImg.append(Imgpath)

                # print(pointindex )

                # print(maskpath)
                trainMask.append(maskpath)

                trainBack.append(backpath)
                # print(Imgpath)
                # print(maskpath)
                # print(pointfiles[i])
                # print(backpath)
        catpointdir = glob.glob(valpointdir[catindex] + '/*')
        catpointdir = sorted(catpointdir)
        for j in range(int(len(catpointdir))):
            valpointfiles = glob.glob(catpointdir[j] + '/*')
            for k in range(int(len(valpointfiles))):

                base = os.path.basename(valpointfiles[k])
                filename = os.path.splitext(base)[0]
                Imgpath = '/home/su_j615/out_focusing/val2017/' + filename[0:12] +'.jpg'
                maskpath = valpointfiles[k].replace('points10', 'masks')
                valpointindex = maskpath.split('/')[6]
                maskpath = maskpath.replace('/' + valpointindex, '')
                backpath = valpointfiles[k].replace('points10', 'backpoints10')
                if (not os.path.isfile(Imgpath) or not os.path.isfile(maskpath) or not os.path.isfile(backpath)):
                    continue
                validPoint.append(valpointfiles[k])
                validImg.append(Imgpath)
                validMask.append(maskpath)
                validBack.append(backpath)
                # print(Imgpath)
                # print(maskpath)
                # print(valpointfiles[k])
                # print(backpath)

    print(len(trainImg))
    print(len(validImg))

    print('start')
    training_generator = DataGenerator(trainPoint,trainBack,trainImg, trainMask, **params)
    validation_generator = DataGenerator(validPoint,validBack,validImg, validMask, **params)
    earlystopping = EarlyStopping(monitor='val_loss',  # 모니터 기준 설정 (val loss)
                                  patience=10,  # 10회 Epoch동안 개선되지 않는다면 종료
                                  )
    model = UNet()
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    results = model.fit(training_generator, validation_data=(validation_generator), epochs=150)
    modelname = "point125_10_v1.h5"
    model.save(modelname)

    print('finish')