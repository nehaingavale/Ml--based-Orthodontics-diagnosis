import cv2
import os
import sys
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN
import uuid
import argparse
import skimage
import colorsys
import tensorflow as tf
import numpy as np
import shutil
import random
import argparse
def get_image(image_path,img_name):
 class TestConfig(Config):
     NAME = "Demo purpose"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 33
 config = TestConfig()



 ROOT_DIR = os.path.abspath("C:/Users/MUNNA/Mask_RCNN/")

 input_image = image_path
 filename = img_name
 output = os.path.join(ROOT_DIR,"static/detections/")


 model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR) #you can insert the any path of this is not important but you need to insert only path 
 model.load_weights('C:/Users/MUNNA/Mask_RCNN/logs/demo/mask_rcnn_demo purpose_0100.h5', by_name=True) #if you run test this script on your local system you need to replace this path 

 class_names = ['BG','11','12','13','14','15','16','17','18','21','22','23','24','25','26','27','28','31','32','33','34','35','36','37','38','41','42','43','44','45','46','47','48']

 def random_colors(N):
     np.random.seed(1)
     colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
     return colors

 colors = random_colors(len(class_names))
 class_dict = {
    name: color for name, color in zip(class_names, colors)
 }

 def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
      image[:, :, n] = np.where(
       mask == 1,
       image[:, :, n] * (1 - alpha) + alpha * c,
       image[:, :, n]
       )
    return image


 def display_instances(image, boxes, masks, ids, names, scores):
     n_instances = boxes.shape[0]
     print("no of potholes in frame :",n_instances)
     if not n_instances:
         print('NO INSTANCES TO DISPLAY')
     else:
         assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

     for i in range(n_instances):
         if not np.any(boxes[i]):
             continue
         y1, x1, y2, x2 = boxes[i]
         label = names[ids[i]]
         color = class_dict[label]
         score = scores[i] if scores is not None else None
         caption = '{} {:.2f}'.format(label, score) if score else label
         random_name = str(uuid.uuid4())
         mask = masks[:, :, i]  
         #mask1 = np.zeros(image.shape, np.uint8)
         image = apply_mask(image, mask, color)
         image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
         #mask1= cv2.polylines(mask1, [mask], True, (255, 255, 255), 2)
         #Mask2 = cv2.fillPoly(mask1.copy(), [points], (255, 255, 255)) 
         image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        
     return image

 frame = cv2.imread(input_image)
 results = model.detect([frame], verbose=0)
 r = results[0]
 masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
 random_name = str(uuid.uuid4())
 #cv2.imwrite(output + "/" + random_name + ".jpg", masked_image) 
 cv2.imwrite(output + '{}' .format(filename), masked_image)
 print('output saved to: {}'.format(output +  '{}'.format(filename)))
