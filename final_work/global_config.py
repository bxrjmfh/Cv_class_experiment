import os
import sys
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
IMAGE_DIR_NP = os.path.join(ROOT_DIR,"np_data/train_data/images.pkl")
IMAGE_DIR_NP_U = os.path.join(ROOT_DIR,"np_data/useful_detail/images.pkl")
EDGE_DIR_NP = os.path.join(ROOT_DIR,"np_data/outline_result/deges.pkl")
EDGE_DIR_NP_U = os.path.join(ROOT_DIR,"np_data/useful_detail/deges.pkl")
GT_MASK_NP = os.path.join(ROOT_DIR,"np_data/train_data/masks.pkl")
GT_MASK_NP_U = os.path.join(ROOT_DIR,"np_data/useful_detail/masks.pkl")
ID_NP = os.path.join(ROOT_DIR,"np_data/train_data/class_ids.pkl")
ID_NP_U = os.path.join(ROOT_DIR,"np_data/useful_detail/class_ids.pkl")
OUT_MASK_NP = os.path.join(ROOT_DIR,'np_data/out_mask/out_mask.pkl')
OUT_MASK_NP_U = os.path.join(ROOT_DIR,'np_data/useful_detail/out_mask.pkl')
# pure mask result , and merge the same class
OUT_MASK_NP_PARTI = os.path.join(ROOT_DIR,'np_data/out_mask/out_mask{}_{}.pkl')
OUT_MASK_NP_PATH = os.path.join(ROOT_DIR,'np_data/out_mask')
SPLIT_PATH = os.path.join(ROOT_DIR,'np_data/split_data')
# 存储分块后的细节数据
MODEL_PATH = os.path.join(ROOT_DIR,'model2')
# 存储模型路径

# values
block_size = 15
# 图像分块的大小


# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))