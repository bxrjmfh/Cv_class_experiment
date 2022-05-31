# 本文件加载已经训练好的mask
import os
import sys
import random
import numpy as np
import skimage.io

# 加载必要包
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize
# 加载MASK_RCNN

sys.path.append(os.path.join(ROOT_DIR, 'coco/'))
import coco
# 加载coco数据集训练出来的必要数据

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# 显示配置

from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# model.load_weights(COCO_MODEL_PATH, by_name=True)

import tensorflow.compat.v1 as tf

tf.keras.Model.load_weights(model.keras_model, COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
import pickle
from global_config import IMAGE_DIR_NP,OUT_MASK_NP_PARTI
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")
with open(IMAGE_DIR_NP, 'rb') as infile:
    images = pickle.load(infile)
# Run detection
def out_masks():
    batch_size = 10
    start_point = 270
    for i in range(start_point, len(images), batch_size):
        results = []
        logging.info("processing batch {}".format(i))
        for j in range(i, min(i + batch_size, len(images)), 1):
            logging.info("processing {}/{}".format(j, len(images)))
            results += model.detect([images[j]], verbose=1)
        with open(OUT_MASK_NP_PARTI.format(i, i + batch_size - 1), 'wb') as outfile:
            pickle.dump(results, outfile, pickle.HIGHEST_PROTOCOL)
            logging.info("successfully saved batch {}".format(i / batch_size))

# out_masks()

def process_test_pic(images):
    flag = 0
    while flag == 0:
        img = images[np.random.randint(0, len(images))]
        result = model.detect([img],verbose=1)[0]
        if len(result["class_ids"])==1:
            return img,class_names[result["class_ids"][0]],result['masks']
        else:
            continue


# # Visualize results
# image = images[129]
# r = model.detect([image], verbose=1)[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
# DATA_PATH = ROOT_DIR+'/raw_mask_data/'+file_names[0]
# if '.' in DATA_PATH:
#     DATA_PATH=DATA_PATH.replace('.','_')
# if not os.path.exists(DATA_PATH):
#     os.mkdir(DATA_PATH)
#
# np.save(DATA_PATH+'/rois.npy',r['rois'])
# np.save(DATA_PATH+'/class_ids.npy',r['class_ids'])
# np.save(DATA_PATH+'/scores.npy',r['scores'])
# np.save(DATA_PATH+'/masks.npy',r['masks'])
# # 存储兴趣区域，模板以及分数类别等

