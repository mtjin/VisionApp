
import os
from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import blur_process_funcs
import time
from imageprocessing import *

import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)


name ='.000000081988.jpg'
maskname = 'mask.png'
result=cv2.imread(name, cv2.IMREAD_COLOR)
cv2.imwrite('result.png',result)
img = cv2.imread('result.png', cv2.IMREAD_COLOR)
temp = np.full((65, 65), 255)
foreground = []
background = []
model = load_model('point65_v4.h5')


def createPointLabel(pointlist, feature):
    a = len(img)
    b = len(img[0])
    patch = np.zeros((a,b))
    kernel1d = cv2.getGaussianKernel(65, 7)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    gaussianPath = temp * kernel2d
    k = 255 / gaussianPath[32][32]
    gaussianPath = k * gaussianPath
    for i in range(len(pointlist)):
        patchY = pointlist[i][1]
        patchX = pointlist[i][0]
        for j in range(patchY - 32, patchY + 33):
            for k in range(patchX - 32, patchX + 33):
                if (j < 0 or k < 0 or j > a - 1 or k > b - 1):
                    continue
                patch[j][k] = max(patch[j][k],gaussianPath[j - (patchY - 32)][k - (patchX - 32)])
        name = feature + '.png'
        cv2.imwrite(name,patch)

    return patch
def boundingbox(pointlist):
    a = len(img)
    b = len(img[0])
    patcha = np.zeros((a, b))
    print(pointlist)
    xlist = []
    ylist = []
    for i in range(len(pointlist)):
        xlist.append(pointlist[i][0])
        ylist.append(pointlist[i][1])

    nx = min(xlist)
    mx = max(xlist)
    ny = min(ylist)
    my = max(ylist)
    for j in range(ny - 62, my + 62):
        for k in range(nx - 62, mx + 62):
            if (j < 0 or k < 0 or j > a - 1 or k > b - 1):
                continue
            patcha[j][k] = 255
    cv2.imwrite('bounding.png', patcha)
    return patcha
def modelinput():
    Input = np.zeros((1, 288, 480, 5))
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    Point = cv2.imread('foreground.png', cv2.IMREAD_GRAYSCALE)
    dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    back = cv2.imread('background.png', cv2.IMREAD_GRAYSCALE)
    dstback = cv2.resize(back, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    # bounding = cv2.imread('bounding.png', cv2.IMREAD_GRAYSCALE)
    # dstbounding = cv2.resize(bounding, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    vimg = np.array(dstimg)
    valid_Point = np.array(dstPoint)
    valid_back = np.array(dstback)
    Input[0, :, :, 0:3] = vimg
    Input[0, :, :, 3] = valid_Point
    Input[0, :, :, 4] = valid_back
    # Input[0, :, :, 5] = dstbounding
    Input = Input.reshape(1, 288, 480, 5).astype('float32') / 255.0
    return Input

# 마우스 콜백 함수: 연속적인 원을 그리기 위한 콜백 함수
def DrawConnectedCircle(event, x, y, flags, param):
    global drawing
    # 마우스 왼쪽 버튼이 눌리면 드로윙을 시작함
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        foreground.append([x,y])
        print('foreground')
        print(x, y)

    # 마우스 왼쪽 버튼을 떼면 드로윙을 종료함
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    # 마우스 오른쪽 버튼이 눌리면 드로윙을 시작함
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        background.append([x,y])
        print('background')
        print(x, y)


    # 마우스 오른쪽 버튼을 떼면 드로윙을 종료함
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False

drawing = False
cv2.namedWindow('image')
cv2.setMouseCallback('image', DrawConnectedCircle)
cur_char = -1
prev_char = -1
while (1):
    frame = cv2.imshow('image', img)
    c = cv2.waitKey(1)
    if c == 27:
        break
    if c > -1 and c != prev_char:
        cur_char = c
    prev_char = c
    if cur_char == ord('s'):
        print('s')
        cur_char = -1
        createPointLabel(foreground,'foreground')
        createPointLabel(background,'background')
        # boundingbox(foreground)
        Input = modelinput()
        preds = model.predict(Input)
        pred = preds[0].astype('float32') * 255
        cv2.imwrite(maskname, pred)
        mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=(len(img[0]), len(img)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(maskname, mask)
        cv2.imwrite(maskname,make_mask())
        mask = cv2.imread(maskname)
        image = cv2.imread('result.png')

        # blur_image = blur_process_funcs.apply_blur_(np.array(image), np.array(mask))
        # cv2.imwrite('blur.png', blur_image)
        # blur = cv2.imread('blur.png', cv2.IMREAD_COLOR)
        # cv2.imshow('blur', blur)
        a = 0.5
        b = 1.0 - a
        dstmask = cv2.addWeighted(mask, a, image, b, 0)
        cv2.imshow('mask', dstmask)
        cv2.waitKey(1)
    elif cur_char == ord('b'):
        mask = cv2.imread(maskname,0)
        target = cv2.imread('target.png')
        blur = cv2.GaussianBlur(image, (5,5), 5)
        blur = combination(blur,target,mask)
        cv2.imwrite('outfocusing.png', blur)
        cv2.imshow('dst', blur)
        cv2.waitKey(1)
    else:
        output = frame

cv2.destroyAllWindows()



