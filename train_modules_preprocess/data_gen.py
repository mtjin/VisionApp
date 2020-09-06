import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(vertical_flip=True,horizontal_flip=True,zoom_range=0.1, rotation_range=0.4,width_shift_range=0.1,height_shift_range=0.1)

sample_x = np.zeros((1,480, 854, 3))
sample_y = np.zeros((1,480, 854,1))
sample_img = cv2.imread('sample_img.jpg')
sample_mask = cv2.imread('sample_mask.png')[:,:,0]
sample_x[0] = sample_img
sample_y[0,:,:,0] = sample_mask

lx =[]
ly =[]
count = 0
for batch in train_gen.flow(sample_x,seed=2019):
    #print(batch[0].shape)
    lx.append(batch[0])
    count+=1
    if(count==600):
        break

count = 0    
for batch in train_gen.flow(sample_y,seed=2019):
    ly.append(batch[0])
    count+=1
    if(count==600):
        break
