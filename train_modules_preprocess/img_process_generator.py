import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras_preprocessing.image import image_data_generator


trainPointNames=[]
for dirname, _, filenames in os.walk('./Annotations/Point'):
    for filename in filenames:
           trainPointNames.append(os.path.join(dirname, filename))

#Train
trainMask = []
trainImg = []
trainPoint = []
trainPointNames.sort()
for j in range(len(trainPointNames)):
    trainPoint.append(trainPointNames[j])
    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')
    Imgpath = Imgpath.replace('/Point', '/480p')
    Imgpath = Imgpath.replace('.png','.jpg')
    trainImg.append(Imgpath)
    maskpath = trainPointNames[j].replace('/Point','/480p')
    trainMask.append(maskpath)
    
#valid 
validMask = []
validImg = []
validPoint = []
validNPoint = []

    


for j in range(len(trainPointNames)):
    validPoint.append(trainPointNames[j])
    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')
    Imgpath = Imgpath.replace('/Point', '/480p')
    Imgpath = Imgpath.replace('.png', '.jpg')
    validImg.append(Imgpath)
    maskpath = trainPointNames[j].replace('/Point', '/480p')
    validMask.append(maskpath)



sample_img = cv2.imread('sample_img.jpg')
sample_mask = cv2.imread('sample_mask.png')[:,:,0]

def get_area(x,y,endx,endy):
    lx = np.abs(endx-x)
    ly = np.abs(endy-y)
    return ly*lx

def get_coner_points_of_img(mask):
    max_x = -9999
    max_y = -9999
    
    min_x = 9999
    min_y = 9999
    
    for i,row in enumerate(mask):
        
        for j,col in enumerate(row):
            if(col==0):
                continue
            if(max_x<i):
                max_x =i
            if(min_x>i):
                min_x =i
            if(max_y<j):
                max_y =j
            if(min_y>j):    
                min_y =j
    return max_x,max_y,min_x,min_y
    
    
max_x,max_y,min_x,min_y =get_coner_points_of_img(sample_mask)

def copy_portion_of_object(img,mask):
    height,width,_ = img.shape
    bmax_x,bmax_y,bmin_x,bmin_y  = get_coner_points_of_img(mask)

    imin_x =0
    imin_y =0
    imax_x = height
    imax_y = width
    
    res_img = img
    b_area = get_area(bmax_x,bmax_y,bmin_x,bmin_y)
    
    copy_target = img[bmin_x:bmax_x,bmin_y:bmax_y,:]
    copy_distnace_x = bmax_x-bmin_x
    copy_distnace_y = bmax_y-bmin_y
    
    top_x_dis = np.abs(bmin_x-imin_x)
    top_y_dis = np.abs(bmax_y-imin_y)
    btm_x_dis = np.abs(imax_x-bmin_x)
    btm_y_dis = np.abs(imax_y-bmax_y)
    if(top_y_dis-10==9989 or top_x_dis-10==9989 ):
        return res_img
    if(np.abs(imin_x-top_x_dis) >20 and np.abs(imin_y-top_y_dis) > 20):
        res_img[imin_x:imin_x+top_x_dis-10,imin_y:imin_y+top_y_dis-10]= cv2.resize(copy_target,(top_y_dis-10,top_x_dis-10))
    
    top_x_dis=np.abs(imax_x-top_x_dis)
    top_y_dis=np.abs(imax_y-top_y_dis)
    if((top_x_dis) >20 and (top_y_dis) > 20):
        res_img[imin_x:imin_x+top_x_dis-10,bmax_y:bmax_y+top_y_dis-10]= cv2.resize(copy_target,(top_y_dis-10,top_x_dis-10))
    
    return res_img

train_x = np.zeros((len(trainImg),288,480, 4))
valid_x = np.zeros((len(validImg),288,480, 4))
train_mask_y = np.zeros((len(trainMask),288,480))
valid_mask_y = np.zeros((len(validMask),288,480))

for index in range(len(validImg)):
    if "G" in validMask[index]:
        continue
    mask = cv2.imread(validMask[index], cv2.IMREAD_GRAYSCALE)
    dstmask = cv2.resize(mask, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    img = cv2.imread(validImg[index], cv2.IMREAD_COLOR)
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    Point = cv2.imread(validPoint[index], cv2.IMREAD_GRAYSCALE)
    dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    vmask = np.array(dstmask)
    vimg = np.array(dstimg)
    valid_Point =np.array(dstPoint)
    valid_x[index, :, :, 0:3] = vimg
    valid_x[index, :, :, 3] = valid_Point
    valid_mask_y[index,:,:] = vmask   


for j in range(len(trainPointNames)):
    Annopath = trainPointNames[j]
    Annopath = Annopath.replace('/Point', '/480p')
    Annopath = Annopath.replace('.png','.jpg')
    Annopath = Annopath.split('\\')
    temp_path = ""
    for one in Annopath:
        if one == Annopath[-1]:
            temp_path+= "G"+one
        else:
            temp_path +=one+'\\'
    Annopath =temp_path
    cv2.imwrite(Annopath,valid_mask_y[j])
    
    Imgpath = trainPointNames[j].replace('/Annotations', '/JPEGImages')
    Imgpath = Imgpath.replace('/Point', '/480p')
    Imgpath = Imgpath.replace('.png','.jpg')
    Imgpath = Imgpath.split('\\')
    temp_path = ""
    for one in Imgpath:
        if one == Imgpath[-1]:
            temp_path+= "G"+one
        else:
            temp_path +=one+'\\'
    g_img = (copy_portion_of_object(np.uint8(valid_x[j,:,:,0:3]),valid_mask_y[j]))
    cv2.imwrite(temp_path,g_img)
    



