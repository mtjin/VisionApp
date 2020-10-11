import os
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model

def discard():
    base_preict_path = './model_predicts'
    predicted_list = getPredict_List();
    for one in predicted_list:
        tmp_path = os.path.join(base_preict_path,one)
        os.remove(tmp_path)
    

def load_Model(path):
    model = load_model(path)
    return model

def get_img_mask(img,user_anno,model):
    return model.predict(img,user_anno)


def getPredict_List():
    p_list =  os.listdir('./model_predicts')
    return p_list


def remove_annoby():
    None
    
p1 = cv2.imread('./pr_h1.png')[:,:,0]
p1 = cv2.resize(p1,((480,854)))
p2 = cv2.imread('./label_h1.png')[:,:,0]
p1 = cv2.resize(p1,((480,854)))


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

    
    
    
    
    
    