#!/usr/bin/env python
# coding: utf-8

# In[302]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


# In[217]:


def path_join(p1,p2):
    return os.path.join(p1,p2)


# In[243]:


def compare_ids(img,anns):
    if(anns['image_id']==img['id']):
        return True
    return False


# In[218]:


#json_file_path ='/Users/jeongseong-ug/Documents/Programming/CNU_ITS/COCO/annotations'
#val_2017_path ='/Users/jeongseong-ug/Documents/Programming/CNU_ITS/COCO/val2017'

dataDir='./'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)


# In[219]:


coco=COCO(annFile)


# In[220]:


cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# In[221]:


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [324158])
imgs = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])
img = imgs[0]


# In[222]:


categories = 'person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball glove skateboard surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed dining table toilet tv laptop mouse remote keyboard cell phone microwave oven toaster sink refrigerator book clock vase scissors teddy bear hair drier toothbrush'.split(' ')


# In[224]:


# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()


# In[145]:


plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)


# In[149]:


# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)


# In[ ]:





# In[307]:


def get_point_ann(img,ann):
    max_x=-9999
    max_y=-9999
    min_x=9999
    min_y=9999
    if(not compare_ids(img,ann)): 
        return ;
    mask_val = coco.annToMask(ann)
    for x,row in enumerate(mask_val):
        if (x<min_x):
                min_x=x
        if (x>max_x):
                max_x=x
        for y,col in enumerate(row):
            if(col==0):
                continue
            if (y<min_y):
                min_y=y
            if (y>max_y):
                max_y=y
    center_x =np.uint32((max_x+min_x)/2)
    center_y =np.uint32((max_y+min_y)/2)
    
    point_map = np.zeros((img['height'],img['width']))
    point_map[center_x-1:center_x+1,center_x-2:center_x+2]=255
    point_map[center_x-7:center_x+7,center_y-7:center_y+7]= cv2.GaussianBlur(point_map[center_x-7:center_x+7,center_y-7:center_y+7],(7,7),40)
    return point_map
                
                
        
        


# In[308]:


wwww =get_point_ann(img,anns[0])
plt.imshow(wwww)


# In[289]:


# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)


# In[272]:


test_seg_info= image_file_anno_info[0]['segmentation']


# In[ ]:


np.asarray(test_seg_info)


# In[293]:


img


# In[ ]:




