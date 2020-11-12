
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib
import pylab
import cv2
import os

def change(image):
    x = len(image[0])
    y = len(image)
    pointlist = []
    for i in range(y):
        for j in range(x):
            if(100 > image[i][j]):
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_ 0"] = "4"
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    dataDir='/home/su_j615/out_focusing'
    dataType='train2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    maskDir='/home/su_j615/out_focusing/trainmasks/'
    coco=COCO(annFile)



    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print(len(nms))

    for i in range(len(nms)):
        catIds = coco.getCatIds(catNms=[nms[i]])
        imgIds = coco.getImgIds(catIds=catIds)


        dirpath = maskDir + nms[i] + '/'
        print(dirpath)
        if not (os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
        imglen = len(imgIds)
        if(imglen > 300):
            imglen = 300
        for j in range(2,imglen):
            img = coco.loadImgs(imgIds[j])[0]
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

            mask = coco.annToMask(anns[0])
            for k in range(len(anns)):
                maskfile_path = dirpath + img['file_name']
                maskfile_path = maskfile_path.replace(".jpg", "")
                maskfile_path = maskfile_path + str(k) + '.png'
                if not (os.path.isfile(maskfile_path)):
                    mask = coco.annToMask(anns[k])
                    matplotlib.image.imsave(maskfile_path, mask)
                    chimg = cv2.imread(maskfile_path, cv2.IMREAD_GRAYSCALE)
                    changeimg = change(chimg)
                    cv2.imwrite(maskfile_path, changeimg)
                    chimg = cv2.imread(maskfile_path, cv2.IMREAD_GRAYSCALE)
                    reshapeimg = cv2.resize(chimg, dsize=(854, 480), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(maskfile_path, reshapeimg)


