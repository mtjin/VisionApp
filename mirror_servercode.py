
import flask
from flask import request, render_template
from flask import send_file
import os
from PIL import Image
import numpy as np
from utils import *
from keras.models import load_model
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)
model = load_model("point65_v2.h5")
app = flask.Flask(__name__)

def modelinput(img):
    Input = np.zeros((1, 288, 480, 5))
    dstimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    Point = cv2.imread('PP.png', cv2.IMREAD_GRAYSCALE)
    dstPoint = cv2.resize(Point, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    back = cv2.imread('NP.png', cv2.IMREAD_GRAYSCALE)
    dstback = cv2.resize(back, dsize=(480, 288), interpolation=cv2.INTER_AREA)
    vimg = np.array(dstimg)
    valid_Point = np.array(dstPoint)
    valid_back = np.array(dstback)
    Input[0, :, :, 0:3] = vimg
    Input[0, :, :, 3] = valid_Point
    Input[0, :, :, 4] = valid_back
    Input = Input.reshape(1, 288, 480, 5).astype('float32') / 255.0
    return Input

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
        img = cv2.imread(file_dir, cv2.IMREAD_COLOR)

        ori_x = img.shape[0]
        ori_y = img.shape[1]

        rimg = cv2.resize(img, dsize=(480, 288), interpolation=cv2.INTER_AREA)

        # 내가 찍은 X, Y 좌표 리스트 (Float 형)
        x_list = []
        y_list = []
        for value in request.form.getlist('x'):
            x_list.append(int(float(value)))
        for value in request.form.getlist('y'):
            y_list.append(int(float(value)))

        PP = get_User_Annotation_point(x_list,y_list, img)
        cv2.imwrite('PP.png', PP)

        # plt.imshow(PP)
        # plt.show()
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
        cv2.imwrite('NP.png', NP)

        modelinput(img)
        preds = model.predict(input_data)
        print("끝")
        maskfile = 'mask.png'
        pred = preds[0].astype('float32') * 255
        cv2.imwrite(maskfile, pred)
        mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=(len(img[0]), len(img)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(maskfile, mask)
        mask = Image.open(maskfile)
        image = Image.open(maskfile)
        # img1 사진은 점점 투명해지고 img2 사진은 점점 불투명해짐
        a = 0.5
        b = 1.0 - a
        dst = cv2.addWeighted(mask, a, img, b, 0)
        cv2.imwrite('maskresult.jpg', dst)

    return send_file('maskresult.jpg', mimetype='image/jpg')
    # return "성공"


# 실행에서 메인 쓰레드인 경우, 먼저 모델을 불러온 뒤 서버를 시작합니다.
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))


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
