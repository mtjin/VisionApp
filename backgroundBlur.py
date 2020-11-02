import numpy as np
import cv2
from PIL import Image
def target_remove(img, mask):
    remove_img = img
    target_img = np.zeros(img.shape)
    print(img.shape)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j]> 200:
               target_img[i, j, :] = img[i, j, :]
               remove_img[i,j,:] = 0
               mask[i][j] = 255
            else:
               mask[i][j] = 0
    return target_img, remove_img, mask

def combination(background, target, mask):
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j]> 200:
               background[i,j,:] = target[i,j,:]
    return background


