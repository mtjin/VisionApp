import os
from PIL import Image
import numpy as np
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