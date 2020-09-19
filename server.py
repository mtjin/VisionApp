# -*- coding: utf-8 -*- 


import os
#import cv2
from flask import jsonify

import numpy as np
import flask
from flask import request, render_template
import io
import redis
import tensorflow as tf
import matplotlib

#from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.datastructures import FileStorage

from blur_process_funcs import *
from utils import *
#model_path = './model/unet_nomal_ver4.h5'
# Flask 애플리케이션과 Keras 모델을 초기화합니다.
#import redis
#redis_conn =  redis.StrictRedis(host='localhost',port=6379,db=0)
#구현한 유틸리티 import
app = flask.Flask(__name__)
#CORS(app)


#model = load_Model(model_path);


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     sel = ("Hi")
#     return sel
#
# @app.route("/outfocus", methods=['POST'])
# def get_img_mask():
#     join_source = json.loads(request.data.decode("utf-8"))
#     None
#
# @app.route("/imgsave", methods=['POST'])
# def imgsave():
#     join_source = json.loads(request.data.decode("utf-8"))
#     None
# @app.route("/imgdisgard", methods=['POST'])
# def imgdisgard():
#     join_source = json.loads(request.data.decode("utf-8"))
#     None
   

# 파일 업로드
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(flask.request.files)
        print(flask.request.files.get('image'))
        #print(flask.request.files['20200912_174631.jpg'])
        print(flask.request.form)
        # 내가 찍은 X, Y 좌표 리스트 (Float 형)
        x_list = request.form.getlist('x')
        y_list = request.form.getlist('y')
        print(x_list)
        print(y_list)
        # # 처음 그린 좌표 하나만 세팅
        x_first = x_list[0]
        y_first = y_list[0]
        print(x_first)
        print(y_first)
        # 파일 받기
        f = request.files['image']
        f2 =flask.request.files.get('image')
        f2.save('D:\Git\zzzzzzzzz\VisionApp\est.jpg')
        filename = secure_filename(f.filename)
        f.save(os.path.join('./'+filename))
        print(f.filename)
        sfname = str(secure_filename(f.filename))
        f.save(sfname)
    return "성공"


# 실행에서 메인 쓰레드인 경우, 먼저 모델을 불러온 뒤 서버를 시작합니다.
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    #load_model()
    # 0.0.0.0 으로 해야 같은 와이파에 폰에서 접속 가능함
    app.run(host='0.0.0.0')