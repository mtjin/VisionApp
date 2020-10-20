import numpy as np
import cv2
from PIL import Image
def target_remove(img, mask):
    remove_img = img
    target_img = np.zeros(img.shape)
    print(img.shape)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if mask[i][j]> 220:
               target_img[i, j, :] = img[i, j, :]
               remove_img[i,j,:] = 0
               mask[i][j] = 255
            else:
               mask[i][j] = 0
    return target_img, remove_img, mask

def combination(background, target, mask):
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j]> 220:
               background[i,j,:] = target[i,j,:]
    return background
# def back_ground(remove_img, mask):
#     neighbor=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
#
#     for i in range(len(remove_img)):
#         for j in range(len(remove_img[0])):
#             if mask[i][j] > 100:
#                 for p in range(len(remove_img[0][0])):
#                     hist = []
#                     for k in range(len(neighbor)):
#                         a = neighbor[k][0]
#                         b = neighbor[k][1]
#                         if sum(remove_img[i+a,i+b,:]) > 0:
#                             hist.append(remove_img[i+a,i+b,p])
#                     remove_img[i,j,p] = np.mean(hist)
#     back_img = remove_img
#     return back_img

mask = Image.open('mask.png')
image = cv2.imread('result.png', cv2.IMREAD_COLOR)
target_img, remove_img, mask = target_remove(np.array(image), np.array(mask))
cv2.imwrite('target.png', target_img)
cv2.imwrite('remove.png', remove_img)
cv2.imwrite('mask.png', mask)
target = cv2.imread('target.png')
remove = cv2.imread('remove.png')
mask = cv2.imread('mask.png',0)

dst = cv2.inpaint(remove,mask,3,cv2.INPAINT_TELEA)

blur = cv2.GaussianBlur(dst, (5,5), 0)

blur = combination(blur,target,mask)

cv2.imshow('dst',blur)

cv2.waitKey(0)
cv2.destroyAllWindows()