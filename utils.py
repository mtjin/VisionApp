import os
from PIL import Image
import numpy as np
import cv2

from tensorflow.keras.models import Model, load_model
from backgroundBlur import *
    
temp = np.full(( 65 , 65 ), 255 )
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


def get_User_Annotation_point(x_list, y_list,img):
    a = len(img)
    b = len(img[0])
    patch = np.zeros((a, b))
    kernel1d = cv2.getGaussianKernel(65, 7)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    gaussianPath = temp * kernel2d
    k = 255 / gaussianPath[32][32]
    gaussianPath = k * gaussianPath
    for i in range(len(x_list)):
        patchY = y_list[i]
        patchX = x_list[i]
        for j in range(patchY - 32, patchY + 33):
            for k in range(patchX - 32, patchX + 33):
                if (j < 0 or k < 0 or j > a - 1 or k > b - 1):
                    continue
                patch[j][k] = max(patch[j][k], gaussianPath[j - (patchY - 32)][k - (patchX - 32)])
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
    back_img = back_ground(remove_img, mask)

    dst = cv2.inpaint(back_img,mask,3,cv2.INPAINT_TELEA)
    
    blur = cv2.GaussianBlur(dst, (5,5), 0)
    
    blur = combination(blur,target_img,mask)
    return blur
    

    
    
    
    
    