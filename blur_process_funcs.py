import cv2
import numpy as np
import matplotlib.pyplot as plt

blur_kernal_level = [5,7,9,11,13,15,21,25]


"""
sample_img =  cv2.imread("./sample_dog.jpg")
sample_bear= cv2.imread("sample_bear.jpg")
sample_bear_mask = cv2.imread("sample_bear_mask.png")

center_x =np.int(sample_bear.shape[0]/2)
center_y =np.int(sample_bear.shape[1]/2)
"""

def get_center_pix_indexes(mask):
    max_x = 0
    min_x = 9999
    max_y = 0
    min_y = 9999
    for i,x in enumerate(mask[:,:,0]):
        for j,y in enumerate(x):
            if y!=0:
                #print(y)
                if i>max_x:
                    max_x = i
                if i<min_x:
                    min_x = i
                    
                if j>max_y:
                    max_y = j
                if j<min_y:
                    min_y = j  
    center_x = np.uint32((max_x+min_x)/2)
    center_y = np.uint32((max_y+min_y)/2)
    return center_x,center_y,min_x,min_y,max_x,max_y

def apply_blur_(input_img,mask):
    max_x =input_img.shape[0]
    max_y =input_img.shape[1]
    img = cv2.copyMakeBorder( input_img, 25, 25, 25, 25, 4)

    res_mask = np.zeros(input_img.shape)
    res_mask[:,:,0] = (input_img[:,:,0]*(mask[:,:,0]/255))
    res_mask[:,:,1] = (input_img[:,:,1]*(mask[:,:,1]/255))
    res_mask[:,:,2] = (input_img[:,:,2]*(mask[:,:,2]/255))

    blur_res = input_img
    c_x,c_y,Mmin_x,Mmin_y,Mmax_x,Mmax_y =get_center_pix_indexes(mask)
    blur_res = cv2.GaussianBlur(input_img,(11,11),40)
   
    entire_filter_tf=False
    print(img.shape)
    xcounter = 0
    ycounter = 0
    end_blur = False
    top_x_medium = np.uint32(Mmin_x/2)
    bottom_x_medium = np.uint32((Mmax_x+max_x)/2)
    top_y = np.uint32(Mmin_y/2)
    bottom_y_medium = np.uint32((Mmax_y+max_y)/2)
    
    blur_res[top_x_medium:top_y,Mmin_y:bottom_y_medium,:] = cv2.GaussianBlur(input_img[top_x_medium:top_y,Mmin_y:bottom_y_medium,:],(7,7),30)
   
    left_x_alpha = np.abs(np.uint32((Mmin_x-c_x)/20))
    left_y_alpha = np.abs(np.uint32((Mmin_y-c_y)/20))
    right_x_alpha = np.abs(np.uint32((Mmax_x-c_x)/20))
    right_y_alpha = np.abs(np.uint32((Mmax_y-c_y)/20))
    blur_res[Mmin_x:Mmax_x,Mmin_y:Mmax_y,:] = cv2.GaussianBlur(input_img[Mmin_x:Mmax_x,Mmin_y:Mmax_y,:],(9,9),20)
    
    for i,x in enumerate(mask[:,:,0]):
        for j,y in enumerate(x):
            if(y==0):
                continue
            else:
                blur_res[i,j,:]=res_mask[i,j,:]
        
    #total_alpha = (x_alpha+y_alpha+20)
    #dx=np.uint32(np.abs(i-c_x)/(max_x*0.1))
    #dy=np.uint32(np.abs(j-c_y)/(max_y*0.1)) 
            
               
        
            

    return blur_res
