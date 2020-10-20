import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import load_model


def change(image):
    x = len(image[0])
    y = len(image)
    pointlist = []
    for i in range(y):
        for j in range(x):
            if(100 > image[i][j]):
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    files = glob.glob('/tmp/pycharm_project_368/Personmask2/*')
    for i in range(len(files)):
        print(i)
        img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        changeimg = change(img)
        cv2.imwrite(files[i], changeimg)
        img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        reshapeimg = cv2.resize(img, dsize=(854, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(files[i], reshapeimg)
print('finish')