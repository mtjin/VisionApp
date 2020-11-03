# -*- coding: utf-8 -*-

# from google.cloud import storage
import os
# import cv2
from flask import jsonify

import numpy as np
import flask
from flask import request, render_template
from flask import send_file
import io
from PIL import Image
from utils import *
import keras


# from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.datastructures import FileStorage

# from blur_process_funcs import *
#from utils import *

# model_path = './model/unet_nomal_ver4.h5'
# Flask 애플리케이션과 Keras 모델을 초기화합니다.
# import redis
# redis_conn =  redis.StrictRedis(host='localhost',port=6379,db=0)
# 구현한 유틸리티 import
app = flask.Flask(__name__)


# CORS(app)


# model = load_Model(model_path);

# 파일 업로드
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(flask.request.files)
        print(flask.request.files.get('image'))
        # print(flask.request.files['20200912_174631.jpg'])
        print(flask.request.form)
        file_dir = "D:\Git\VisionApp\ex1234.jpg"

        f2 = flask.request.files.get('image')
        f2.save(file_dir)
        # img = cv2.imread('D:\Git\VisionApp\ex1234.jpg')
        # 내가 찍은 X, Y 좌표 리스트 (Float 형)
        #x_list = np.uint(request.form.getlist('x'))
        #y_list = np.uint(request.form.getlist('y'))
        # 유저 백그라운드 포인트

        """서버에서 positive_anno_mask을 upload_file에 대한 리스폰스값으로 보내야댐"""
        # positive_anno_mask = get_User_Annotation_point_Mask(x_list, y_list, img)

        # nx_list = np.uint(request.form.getlist('nx'))
        # ny_list = np.uint(request.form.getlist('ny'))
        # negative_anno_mask = get_User_Annotation_point_Mask(x_list,y_list,img)

        # 파일 받기
        f = request.files['image']

        filename = secure_filename(f.filename)
        f.save(os.path.join('./' + filename))
        print(f.filename)
        sfname = str(secure_filename(f.filename))
        f.save(sfname)
    return send_file(file_dir, mimetype='image/jpg')
    #return "성공"


# 실행에서 메인 쓰레드인 경우, 먼저 모델을 불러온 뒤 서버를 시작합니다.
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    # load_model()
    # 0.0.0.0 으로 해야 같은 와이파에 폰에서 접속 가능함
    app.run(host='0.0.0.0')


# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     # bucket_name = "your-bucket-name"
#     # source_file_name = "local/path/to/file"
#     # destination_blob_name = "storage-object-name"
#
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print(
#         "File {} uploaded to {}.".format(
#             source_file_name, destination_blob_name
#         )
#     )
