import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from backgroundBlur import *

def discard():
    base_preict_path = './model_predicts'
    predicted_list = getPredict_List();
    for one in predicted_list:
        tmp_path = os.path.join(base_preict_path,one)
        os.remove(tmp_path)
    

def load_Model(path):
    model = load_model(path)
    return model

def targetPixelList(image):
    x = len(image[0])
    y = len(image)
    pointlist = []
    for i in range(y):
        for j in range(x):
            if(image[i][j] > 100):
                pointlist.append([i,j])
    return pointlist


def createPointLabel(TPL, image):
    y = len(image)
    x = len(image[0])
    patch = np.zeros((y,x))
    # pointpatch = pointPatch
    temp = np.full(( 65 , 65 ), 255 )
    if(len(TPL)>0):
        index = randint(0,len(TPL)-1)
        kernel1d = cv2.getGaussianKernel(65, 20)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())

        gaussianPath = temp * kernel2d
        k = 255 / gaussianPath[32][32]
        gaussianPath = k*gaussianPath
        patchY = TPL[index][0]
        patchX = TPL[index][1]
        for j in range(patchY-32, patchY+33):
            for k in range(patchX-32, patchX+33):
                if(j<0 or k<0 or j > y-1 or k>x-1):
                    continue
                patch[j][k] = gaussianPath[j-(patchY-32)][k-(patchX-32)]
    return patch


def get_User_Annotation_point_Mask(x_points,y_points,ori_img):
    gauss_kernal = 65
    gauss_gamma = 20
    y = len(ori_img)
    x = len(ori_img[0])
    patch = np.zeros((y,x))
    temp = np.full(( 65 , 65 ), 255 )
    for i,x_ in enumerate(x_points):
        kernel1d = cv2.getGaussianKernel(65, 20)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())

        gaussianPath = temp * kernel2d
        k = 255 / gaussianPath[32][32]
        gaussianPath = k*gaussianPath
        patchY = y_points[i]
        patchX = x_
        for j in range(patchY-32, patchY+33):
            for k in range(patchX-32, patchX+33):
                if(j<0 or k<0 or j > y-1 or k>x-1):
                    continue
                patch[j][k] = gaussianPath[j-(patchY-32)][k-(patchX-32)]
    return patch
    



def IoU(predicted_mask,mask_lable):

    max_count = np.count_nonzero(mask_lable)
    pp = np.uint8(predicted_mask)
    p1 = np.zeros(pp.shape)
    for i,row in enumerate(pp):
        for j,one in enumerate(row):
            if(one<100):
                continue
            else:
                p1[i][j] = 1
    p2= np.uint8(mask_lable/255)
    res = p1*p2
    res_count = np.count_nonzero(res)
    
    return np.float32(res_count/max_count)

def apply_blur(img,mask):
    target_img, remove_img, mask = target_remove(np.array(img), np.array(mask))

    dst = cv2.inpaint(remove_img,mask,3,cv2.INPAINT_TELEA)
    
    blur = cv2.GaussianBlur(dst, (5,5), 0)
    
    blur = combination(blur,target_img,mask)
    return blur
    

    
    
    
    
    