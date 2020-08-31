import os

from flask import jsonify
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import request, render_template
import io
import tensorflow as tf
import matplotlib
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Flask 애플리케이션과 Keras 모델을 초기화합니다.
app = flask.Flask(__name__)
model = None


def load_model():
    # 미리 학습된 Keras 모델을 불러옵니다(여기서 우리는 ImageNet으로 학습되고
    # Keras에서 제공하는 모델을 사용합니다. 하지만 쉽게 하기위해
    # 당신이 설계한 신경망으로 대체할 수 있습니다.)
    global model
    global graph
    model = ResNet50(weights="imagenet")
    graph = tf.get_default_graph()


def prepare_image(image, target):
    # 만약 이미지가 RGB가 아니라면, RGB로 변환해줍니다.
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    # 입력 이미지 사이즈를 재정의하고 사전 처리를 진행합니다.
    # image = image.resize(target)
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)
    # 처리된 이미지를 반환합니다.
    return image


@app.route("/predict222", methods=['POST'])
def predict():
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    return "SUCCESS"


@app.route('/upload', methods=['GET'])
def load_file():
    print("HI BRO")


@app.route('/', methods=['GET'])
def test():
    print("HI GET")


# 파일 업로드
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(flask.request.files)
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
        # 그걸 그대로 적당한 파일명 줘서 저장 (뭔지랄을 해도 파일 저장된걸 볼 수가 없음 어따 저장되는겨 싯벌)
        f.save(secure_filename(f.filename))
        print(f.filename)
        sfname = str(secure_filename(f.filename))
        f.save(sfname)
    return "성공"


# 실행에서 메인 쓰레드인 경우, 먼저 모델을 불러온 뒤 서버를 시작합니다.
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    # 0.0.0.0 으로 해야 같은 와이파에 폰에서 접속 가능함
    app.run(host='0.0.0.0')