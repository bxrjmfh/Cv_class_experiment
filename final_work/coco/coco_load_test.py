import os
import sys
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

import numpy as np
import matplotlib.pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
NP_DATA = "/Data/Rice_Bowl/Python/Cv_class_experiment-master/final_work/coco/coco_data/numpy_data"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import coco
config = coco.CocoConfig()
COCO_DIR = "/Data/Rice_Bowl/Python/Cv_class_experiment-master/final_work/coco/coco_data"
dataset = coco.CocoDataset()
coco=dataset.load_coco(dataset_dir=COCO_DIR,subset= "val",year=2017,return_coco=True)
# Must call before using the dataset
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
# masks = [dataset.load_mask(id) for id in dataset.image_ids]

masks = []
class_ids = []
images = []
for i in range(dataset.image_ids.shape[0]):
    logging.info("processing {} th data".format(i))
    mask,class_id = dataset.load_mask(i)
    if class_id.shape[0] ==1:
        # 选出只有一类的标签
        logging.info(" {} is sigle classes ".format(i))
        image = dataset.load_image(i)
        masks.append(mask)
        class_ids.append(class_id)
        images.append(image)

import pickle
import numpy as np

with open('../np_data/train_data/images.pkl', 'wb') as outfile:
    pickle.dump(images, outfile, pickle.HIGHEST_PROTOCOL)
with open('../np_data/train_data/masks.pkl', 'wb') as outfile:
    pickle.dump(masks, outfile, pickle.HIGHEST_PROTOCOL)
with open('../np_data/train_data/class_ids.pkl', 'wb') as outfile:
    pickle.dump(class_ids, outfile, pickle.HIGHEST_PROTOCOL)



