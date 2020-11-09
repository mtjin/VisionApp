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

def input(img, pp, bp):
    Input = np.zeros((1, 288, 480, 5))
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    Point = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
    dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    back = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
    dstback = cv2.resize(back, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    vimg = np.array(dstimg)
    valid_Point = np.array(dstPoint)
    valid_back = np.array(dstback)
    Input[0, :, :, 0:3] = vimg
    Input[0, :, :, 3] = valid_Point
    Input[0, :, :, 4] = valid_back
    Input = Input.reshape(1, 288, 480, 5).astype('float32') / 255.0
    return Input


def IoU(predicted_mask, mask_lable):
    max_count = np.count_nonzero(mask_lable)
    pp = np.uint8(predicted_mask)
    p1 = np.zeros(pp.shape)
    for i, row in enumerate(pp):
        for j, one in enumerate(row):
            if (one < 100):
                continue
            else:
                p1[i][j] = 1
    p2 = np.uint8(mask_lable / 255)
    res = p1 * p2
    res_count = np.count_nonzero(res)
    return np.float32(res_count / max_count)

if __name__ == "__main__":
    model = load_model('point65_v2.h5')
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    maskfiles = glob.glob('/home/su_j615/out_focusing/valmasks/*')
    valpointdir = glob.glob('/home/su_j615/out_focusing/valpoints/*')

    # valid
    validMask = []
    validImg = []
    validPoint = []
    validBack = []

    for catindex in range(79):

        catpointdir = glob.glob(valpointdir[catindex] + '/*')
        catpointdir = sorted(catpointdir)
        for j in range(int(len(catpointdir))):
            valpointfiles = glob.glob(catpointdir[j] + '/*')
            for k in range(int(len(valpointfiles))):
                base = os.path.basename(valpointfiles[k])
                filename = os.path.splitext(base)[0]
                Imgpath = '/home/su_j615/out_focusing/val2017/' + filename[0:12] +'.jpg'
                maskpath = valpointfiles[k].replace('points', 'masks')
                valpointindex = maskpath.split('/')[6]
                maskpath = maskpath.replace('/' + valpointindex, '')
                backpath = valpointfiles[k].replace('points', 'backpoints')
                if (not os.path.isfile(Imgpath) or not os.path.isfile(maskpath) or not os.path.isfile(backpath)):
                    continue
                validPoint.append(valpointfiles[k])
                validImg.append(Imgpath)
                validMask.append(maskpath)
                validBack.append(backpath)

    print(len(validImg))
    import csv

    f = open('IOU.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(["idx", "cat", "iou", 'iou>0.5'])

    for i in range(len(validPoint)):
        model_input = input(validImg[i],validPoint[i],validBack[i])
        preds = model.predict(model_input)
        pred = preds[0].astype('float32') * 255
        maskname = validPoint[i].replace('valpoints','valIOUmask')
        cv2.imwrite(maskname, pred)
        predmask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
        predmask = cv2.resize(predmask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        orignal = cv2.imread(validMask[i],cv2.IMREAD_GRAYSCALE)
        orignal = cv2.resize(orignal, dsize=(480, 288), interpolation=cv2.INTER_AREA)
        orignalmaskname = validPoint[i].replace('valpoints', 'orginalmask')
        cv2.imwrite(orignalmaskname, orignal)
        iou_value = IoU(predmask, orignal)
        list = validPoint[i].split('/')
        cat = list[5]
        if iou_value > 0.5:
            wr.writerow([i,cat,iou_value, True])
        else :
            wr.writerow([i, cat, iou_value, False])
f.close()

print('start')
print('finish')