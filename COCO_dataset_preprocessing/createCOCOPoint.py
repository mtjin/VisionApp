import errno
from random import randint

import matplotlib
import numpy as np
import os
import glob
import cv2
import pandas as pd
import PIL.Image as pilimg
from PIL import Image
import matplotlib.image as mpimg

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
def pointPatch():
    ##Just calculate 64*64
    arr = np.zeros((64,64,3), dtype=np.uint8)
    imgsize = arr.shape[:2]
    innerColor = (255, 255, 255)
    outerColor = (0, 0, 0)
    for y in range(imgsize[1]):
        for x in range(imgsize[0]):
            #Find the distance to the corner
            distanceToCenter = np.sqrt((x) ** 2 + (y - imgsize[1]) ** 2)

            #Make it on a scale from 0 to 1innerColor
            distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0])

            #Calculate r, g, and b values
            r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
            g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
            b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)
            # print r, g, b
            arr[y, x] = (int(r), int(g), int(b))
    #rotate and combine
    arr1=arr
    arr2=arr[::-1,:,:]
    arr3=arr[::-1,::-1,:]
    arr4=arr[::,::-1,:]
    arr5=np.vstack([arr1,arr2])
    arr6=np.vstack([arr4,arr3])
    arr7=np.hstack([arr6,arr5])
    arr7 = rgb2gray(arr7)
    return arr7


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

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_ 0"] = "4"
    for i in range(10):
        maskfile = '/Personmask2'
        pointfile = '/Person_Point2/point'+str(i)
        dirNames = '/tmp/pycharm_project_368'+maskfile
        # #print(dirNames)
        # #for i in range():
        # if not (os.path.isdir('/tmp/pycharm_project_368/DAVIS/Annotations/Point')):
        #     os.makedirs(os.path.join('/tmp/pycharm_project_368/DAVIS/Annotations/Point'))
        # for i in range(len(dirNames)):
        fileNames = glob.glob(dirNames + "/*")
        print(len(fileNames))
        for i in range(len(fileNames)):
            print(str(i))
            image = cv2.imread(fileNames[i], cv2.IMREAD_GRAYSCALE)
            TPL = targetPixelList(image)
            patch = createPointLabel(TPL, image)
            dirpath = dirNames.replace(maskfile, pointfile)
            if not (os.path.isdir(dirpath)):
                os.makedirs(os.path.join(dirpath))
            path = fileNames[i].replace(maskfile, pointfile)
            cv2.imwrite(path, patch)

