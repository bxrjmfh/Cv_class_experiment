import pickle
import cv2 as cv
from tensorflow import keras
from load_mask_rcnn import process_test_pic
from preprocess_to_block import Pad_to_shape_normalize, Split_edge_blocks_nor
from global_config import IMAGE_DIR_NP_U, block_size, MODEL_PATH
import numpy as np


# 读取图片，数据加载，输入到原始mask-rcnn 网络中
# 提取边缘，分片识别边缘信息
# 输入到网络中，得到预测结果
# 归位，输出最终图片

def Fill_block_in_place(out_mask, block_indexs, results):
    # 将处理后的结果缝合到原图上
    out_mask = out_mask.astype("float32")
    for i, index in enumerate(block_indexs):
        x = index[1]
        y = index[2]
        out_mask[x * block_size:(x + 1) * block_size,
                 y * block_size:(y + 1) * block_size, 0] = results[i]
    return out_mask

with open(IMAGE_DIR_NP_U, 'rb') as infile:
    images = pickle.load(infile)

# use mask-rcnn load the result
img, class_name, out_mask = process_test_pic(images)
# 处理边缘信息
blurred_img = cv.blur(img, ksize=(5, 5))
med_val = np.median(img)
lower = int(max(0, 0.7 * med_val))
upper = int(min(255, 1.3 * med_val))
edge = cv.Canny(image=img, threshold1=lower, threshold2=upper)
# 归一化且进行图像裁剪,处理为模型的形状
mask_nor = Pad_to_shape_normalize(None, [out_mask], False)
edge_nor = Pad_to_shape_normalize(None, [edge], False)
block_indexs, block_out_masks, block_gt_masks, block_edges = \
    Split_edge_blocks_nor([0], mask_nor, mask_nor, edge_nor, block_size)
model_input_1 = np.asarray(block_out_masks)
model_input_2 = np.asarray(block_edges)
model_input_2 = np.expand_dims(model_input_2, axis=3)
# 添加模型且进行预测

mask_model = keras.models.load_model(MODEL_PATH)
mask_model.load_weights(MODEL_PATH)
# 添加加载权重才可以进行预测
# 得到输出结果:
results = mask_model.predict([model_input_1, model_input_2])
results = results.reshape((results.shape[0], block_size, block_size))
adjust_mask = Fill_block_in_place(out_mask, block_indexs, results)

# 显示图片结果
import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.imshow(img)
plt.title("original pic")
plt.subplot(2,2,2)
plt.imshow(out_mask[...,0])
plt.title("raw mask")
plt.subplot(2,2,3)
plt.imshow(edge)
plt.title("edge info")
plt.subplot(2,2,4)
plt.imshow(adjust_mask[...,0])
plt.title("adjusted mask")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()

