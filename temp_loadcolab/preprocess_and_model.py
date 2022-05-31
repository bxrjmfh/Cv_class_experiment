import numpy as np
# from tensorflow import keras
# import keras
import pickle
from global_config import OUT_MASK_NP_U, EDGE_DIR_NP_U, GT_MASK_NP_U,ID_NP_U,SPLIT_PATH

with open(OUT_MASK_NP_U, 'rb') as infile:
    out_masks = pickle.load(infile)
with open(EDGE_DIR_NP_U, 'rb') as infile:
    edges = pickle.load(infile)
with open(GT_MASK_NP_U, 'rb') as infile:
    gt_masks = pickle.load(infile)
with open(ID_NP_U, 'rb') as infile:
    ids = pickle.load(infile)

# 取最大形状
def Get_max_size(out_masks):
    x_max = 0
    y_max = 0
    for m in out_masks:
        if m.shape[0] > x_max:
            x_max = m.shape[0]
        if m.shape[1] > y_max:
            y_max = m.shape[1]
    return x_max, y_max


# 尺寸归一化（未用）
def Pad_to_shape_normalize(shape, inputs,is_pad):
    if is_pad:
        outputs = []
        if inputs[0].dtype == 'uint8':  # edge infos
            for item in inputs:
                out = np.zeros(shape).astype('float32')
                x_, y_ = np.shape(item)
                out[0:x_, 0:y_] = item.astype('float32') / 255.0
                outputs.append(out)
        else:
            for item in inputs:  # mask infos
                out = np.zeros(shape).astype('float32')
                x_, y_ = np.shape(item)
                out[0:x_, 0:y_] = item.astype('float32')
                outputs.append(out)
        outputs = np.asarray(outputs)
        return outputs
    else:
        outputs = []
        if inputs[0].dtype == 'uint8':  # edge infos
            for item in inputs:
                out = item.astype('float32') / 255.0
                outputs.append(out)
        else:
            for item in inputs:  # mask infos
                out = item.astype('float32')
                outputs.append(out)
        outputs = np.asarray(outputs)
        return outputs

# 对每个数据尺寸归一化
target_size = Get_max_size(out_masks)
nor_out_masks = Pad_to_shape_normalize(target_size, out_masks,False)
nor_edges = Pad_to_shape_normalize(target_size, edges,False)
# 归一化为0-1的边缘响应
nor_gt_masks = Pad_to_shape_normalize(target_size, gt_masks,False)

def Split_edge_blocks_nor(ids,out_masks,gt_masks,edges,block_size):
    # ids : (batch_size)    每个图片的类别
    # out_masks : (batchsize,width,heigth) 输出的mask
    # gtmasks 真实mask
    # edge 图像边缘信息， 和上边的输入信息大小类型一致

    # 指定分割任务每个块的索引最大值。原始mask被分割为wmax*hmax个块，但是不全被存储，只会存储边缘图像
    block_indexs = []
    # 每个块的对应原图信息
    block_out_masks = []
    # 分块输出mask信息
    block_gt_masks = []
    # 分块真实mask信息
    block_edges = []
    # 分块边缘信息

    for i,id in enumerate(ids):
        # i为图像编号，id为该图像对应的类别信息
        # split_times 指定图像分割的块数
        w_max = out_masks[i].shape[0] // block_size
        h_max = out_masks[i].shape[1] // block_size
        for j in range(w_max):
            # j为每个块的行号
            for k in range(h_max):
                # k 为列号
                block_index = [i, j, k]
                # 0：图像编号，1，2：w与h的块间地址
                block_out_mask=out_masks[i][j*block_size:(j+1)*block_size,
                                        k*block_size:(k+1)*block_size]
                block_gt_mask =gt_masks[i][j*block_size:(j+1)*block_size,
                                        k*block_size:(k+1)*block_size]
                if (block_out_mask.sum() == 0) | (block_out_mask.sum() == block_size**2)|(block_gt_mask.sum()==0):
                    continue
                    # 如果全为0，则全为背景部分，如果全为1，则是实例内部
                else:
                    # 不全为0，则为边缘，对应提取边缘所在的块
                    block_indexs.append(block_index)
                    block_out_masks.append(block_out_mask)
                    block_gt_masks.append(gt_masks[i][j*block_size:(j+1)*block_size,
                                        k*block_size:(k+1)*block_size])
                    block_edges.append(edges[i][j * block_size:(j + 1) * block_size,
                                          k * block_size:(k + 1) * block_size])

    print('ok')
    return block_indexs,block_out_masks,block_gt_masks,block_edges

def Check_splid(block_indexs,block_out_masks,block_gt_masks,block_edges,index):
    from matplotlib import pyplot as plt
    print(block_indexs[index])
    plt.imshow(edges[block_indexs[index][0]])
    plt.title("full edge of "+str(block_indexs[index][0]))
    plt.show()

    plt.imshow(block_edges[index])
    plt.title("edges of"+str(block_indexs[index]))
    plt.show()
    plt.imshow(block_out_masks[index])
    plt.title("out mask of" + str(block_indexs[index]))
    plt.show()
    plt.imshow(block_gt_masks[index])
    plt.title("gt mask of" + str(block_indexs[index]))
    plt.show()

block_size = 40
block_indexs,block_out_masks,block_gt_masks,block_edges=\
    Split_edge_blocks_nor(ids,nor_out_masks,nor_gt_masks,nor_edges,block_size)


# 处理数据集为指定格式，划分训练集与测试集
partition = int(len(block_out_masks)*(29/30))
raw_masks = np.asarray(block_out_masks)
raw_masks = np.expand_dims(raw_masks,axis=3)
# 添加第三个轴，维度为（，80，80，1）
train_raw = raw_masks[:partition,...]
valid_raw = raw_masks[partition:,...]

input_edges =np.asarray(block_edges)
input_edges = np.expand_dims(input_edges,axis=3)
# 添加第三个轴，维度为（，80，80，1）
train_edges = input_edges[:partition,...]
valid_edges = input_edges[partition:,...]

ref_masks = np.asarray(block_gt_masks)
ref_masks = np.reshape(ref_masks,(ref_masks.shape[0],block_size**2))
train_ref = ref_masks[:partition,...]
valid_ref = ref_masks[partition:,...]


# from tensorflow import keras
# from tensorflow.keras import optimizers

def Construct_model(block_size):
    from tensorflow import keras
    out_mask_input = keras.layers.Input(shape=(block_size, block_size, 1,))
    edge_input = keras.layers.Input(shape=(block_size, block_size, 1,))
    # 输入边缘信息
    conv_mask = keras.layers.Conv2D(filters=15, kernel_size=3, activation='relu')(out_mask_input)
    # max_pooling_mask = keras.layers.MaxPooling2D(pool_size=2)(conv_mask)
    conv_edge = keras.layers.Conv2D(filters=15, kernel_size=3, activation='relu')(edge_input)
    # max_pooling_edge = keras.layers.MaxPooling2D(pool_size=2)(conv_edge)

    merged = keras.layers.Concatenate(axis=1)([out_mask_input, edge_input])
    # merged = keras.layers.Concatenate(axis=1)([max_pooling_mask, max_pooling_edge])
    conv_main1 = keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu')(merged)
    conv_main2 = keras.layers.Conv2D(filters=10, kernel_size=5, dilation_rate=2)(conv_main1)
    max_pooling1 = keras.layers.MaxPooling2D(pool_size=2)(conv_main2)
    conv_main3 = keras.layers.Conv2D(filters=10, kernel_size=5, dilation_rate=2)(max_pooling1)
    max_pooling2 = keras.layers.MaxPooling2D(pool_size=2)(conv_main3)
    flatter = keras.layers.Flatten(data_format=None)(max_pooling2)
    dense1 = keras.layers.Dense(block_size ** 2, activation=keras.activations.sigmoid)(flatter)


    mask_model = keras.models.Model(inputs=[edge_input, out_mask_input], outputs=dense1)
    return mask_model

mask_model = Construct_model(block_size)
from tensorflow.keras import optimizers
mask_model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
history=mask_model.fit(x=[train_raw,train_edges],y=train_ref,
               validation_data=([valid_raw,valid_edges],valid_ref),
               epochs=200, batch_size=20)
history_2=mask_model.fit(x=[train_raw,train_edges],y=train_ref,
               validation_data=([valid_raw,valid_edges],valid_ref),
               epochs=500, batch_size=50)












