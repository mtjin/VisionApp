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
import cv2
from utils import *
from keras.models import load_model
# from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from werkzeug.datastructures import FileStorage

# from blur_process_funcs import *
# from utils import *

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

        file_dir = "uploadimage.jpg"

        input_data = np.zeros((1, 288, 480, 5))
        f2 = flask.request.files.get('image')
        f2.save(file_dir)
        img = cv2.imread(file_dir)
        print(img.shape[0])
        print(img.shape[1])
        ori_x = img.shape[0]
        ori_y = img.shape[1]
        rimg = cv2.resize(img, (480, 288))
        input_data[0, :, :, :3] = rimg

        # 내가 찍은 X, Y 좌표 리스트 (Float 형)
        x_list = []
        y_list = []
        for value in request.form.getlist('x'):
            x_list.append(int(float(value)))
        for value in request.form.getlist('y'):
            y_list.append(int(float(value)))

        PP = get_User_Annotation_point(x_list,y_list, img)
        cv2.imwrite('PP.jpg', PP)
        PP = cv2.resize(PP, (480, 288))
        # plt.imshow(PP)
        # plt.show()
        input_data[0, :, :, 3] = PP
        """서버에서 positive_anno_mask을 upload_file에 대한 리스폰스값으로 보내야댐"""
        # positive_anno_mask = get_User_Annotation_point_Mask(x_list, y_list, img)

        # nx_list = np.uint(request.form.getlist('nx'))
        # ny_list = np.uint(request.form.getlist('ny'))
        # 내가 찍은 X, Y 좌표 리스트 (Float 형)
        x_list = []
        y_list = []
        for value in request.form.getlist('nx'):
            x_list.append(int(float(value)))
        for value in request.form.getlist('ny'):
            y_list.append(int(float(value)))
        NP = get_User_Annotation_point(x_list,y_list, img)

        cv2.imwrite('NP.jpg', NP)
        NP = cv2.resize(NP, (480, 288))
        # plt.imshow(NP)
        # plt.show()
        input_data[0, :, :, 4] = NP

        mask = model.predict(input_data)[0, :, :, 0]

        # print(mask.shape)
        # print(rimg.shape)
        mask = mask * 255
        mask = np.uint8(mask)
        out_focus_res = apply_blur(rimg, mask)
        out_focus_res = cv2.resize(out_focus_res, (ori_y, ori_x))
        cv2.imwrite('outfocus.jpg', out_focus_res)

        # 파일 받기
        f = request.files['image']

        filename = secure_filename(f.filename)
        f.save(os.path.join('./' + filename))
        print(f.filename)
        sfname = str(secure_filename(f.filename))
        f.save(sfname)

    return send_file('outfocus.jpg', mimetype='image/jpg')
    # return "성공"


# 실행에서 메인 쓰레드인 경우, 먼저 모델을 불러온 뒤 서버를 시작합니다.
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))

    model = load_model('point65_v2.h5')
    # load_model()
    # 0.0.0.0 으로 해야 같은 와이파에 폰에서 접속 가능함
    app.run(host='0.0.0.0', debug=True)

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
