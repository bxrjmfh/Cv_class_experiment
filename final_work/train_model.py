import numpy as np
import pickle
from global_config import OUT_MASK_NP_U, EDGE_DIR_NP_U, GT_MASK_NP_U, ID_NP_U, block_size, MODEL_PATH
from preprocess_to_block import Pad_to_shape_normalize, Split_edge_blocks_nor, Get_max_size


def Construct_model(block_size):
    from tensorflow import keras
    out_mask_input = keras.layers.Input(shape=(block_size, block_size, 1,))
    edge_input = keras.layers.Input(shape=(block_size, block_size, 1,))
    # 输入边缘信息
    conv_mask = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu')(out_mask_input)
    # max_pooling_mask = keras.layers.MaxPooling2D(pool_size=2)(conv_mask)
    conv_edge = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu')(edge_input)
    # max_pooling_edge = keras.layers.MaxPooling2D(pool_size=2)(conv_edge)

    merged = keras.layers.add(inputs=[conv_mask, conv_edge])
    # merged = keras.layers.Concatenate(axis=1)([max_pooling_mask, max_pooling_edge])
    conv_main1 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu')(merged)
    conv_main2 = keras.layers.Conv2D(filters=40, kernel_size=3, dilation_rate=2)(conv_main1)
    # max_pooling1 = keras.layers.MaxPooling2D(pool_size=2)(conv_main2)
    conv_main3 = keras.layers.Conv2D(filters=40, kernel_size=3, dilation_rate=2)(conv_main2)
    # max_pooling2 = keras.layers.MaxPooling2D(pool_size=2)(conv_main3)
    conv_main4 = keras.layers.Conv2D(filters=40,kernel_size=3,activation='relu')(conv_main3)
    flatter = keras.layers.Flatten(data_format=None)(conv_main3)
    dense1 = keras.layers.Dense(block_size ** 2, activation=keras.activations.softmax)(flatter)
    flatter2 = keras.layers.Flatten(data_format=None)(out_mask_input)
    merge2 = keras.layers.add(inputs=[flatter2, dense1])
    dense2 = keras.layers.Dense(block_size ** 2, activation=keras.activations.softmax)(merge2)

    mask_model = keras.models.Model(inputs=[edge_input, out_mask_input], outputs=dense2)
    return mask_model


with open(OUT_MASK_NP_U, 'rb') as infile:
    out_masks = pickle.load(infile)
with open(EDGE_DIR_NP_U, 'rb') as infile:
    edges = pickle.load(infile)
with open(GT_MASK_NP_U, 'rb') as infile:
    gt_masks = pickle.load(infile)
with open(ID_NP_U, 'rb') as infile:
    ids = pickle.load(infile)
# 对每个数据尺寸归一化
target_size = Get_max_size(out_masks)
nor_out_masks = Pad_to_shape_normalize(target_size, out_masks, False)
nor_edges = Pad_to_shape_normalize(target_size, edges, False)
# 归一化为0-1的边缘响应
nor_gt_masks = Pad_to_shape_normalize(target_size, gt_masks, False)

block_indexs, block_out_masks, block_gt_masks, block_edges = \
    Split_edge_blocks_nor(ids, nor_out_masks, nor_gt_masks, nor_edges, block_size)

# 处理数据集为指定格式，打乱数据

raw_masks = np.asarray(block_out_masks)
raw_masks = np.expand_dims(raw_masks, axis=3)
input_edges = np.asarray(block_edges)
ref_masks = np.asarray(block_gt_masks)

randomize = np.arange(len(raw_masks))
np.random.shuffle(randomize)
raw_masks = raw_masks[randomize]
input_edges = input_edges[randomize]
ref_masks = ref_masks[randomize]

# 划分训练集和测试集

partition = int(29 * len(raw_masks) / 30)
train_raw = raw_masks[:partition, ...]
valid_raw = raw_masks[partition:, ...]
# 添加第三个轴，维度为（，80，80，1）
input_edges = np.expand_dims(input_edges, axis=3)
# 添加第三个轴，维度为（，80，80，1）
train_edges = input_edges[:partition, ...]
valid_edges = input_edges[partition:, ...]

ref_masks = np.reshape(ref_masks, (ref_masks.shape[0], block_size ** 2))
train_ref = ref_masks[:partition, ...]
valid_ref = ref_masks[partition:, ...]

mask_model = Construct_model(block_size)
from tensorflow.keras import optimizers

mask_model.summary()

mask_model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-3), metrics=['acc'])

history_2 = mask_model.fit(x=[train_raw, train_edges], y=train_ref,
                           validation_data=([valid_raw, valid_edges], valid_ref),
                           epochs=50, batch_size=5000)

mask_model.save(MODEL_PATH)