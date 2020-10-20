import glob
import os
import cv2


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    files = glob.glob('/tmp/pycharm_project_368/val2017/*')
    for i in range(len(files)):
        print(i)
        img = cv2.imread(files[i], cv2.IMREAD_COLOR)
        reshapeimg=cv2.resize(img, dsize=(854, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(files[i], reshapeimg)

print('finish')