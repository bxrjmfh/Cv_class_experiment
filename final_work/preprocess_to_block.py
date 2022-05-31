import numpy as np
import pickle
from global_config import OUT_MASK_NP_U, EDGE_DIR_NP_U, GT_MASK_NP_U, ID_NP_U, block_size


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
def Pad_to_shape_normalize(shape, inputs, is_pad):
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


def Split_edge_blocks_nor(ids, out_masks, gt_masks, edges, block_size):
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

    for i, id in enumerate(ids):
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
                block_out_mask = out_masks[i][j * block_size:(j + 1) * block_size,
                                 k * block_size:(k + 1) * block_size]
                block_gt_mask = gt_masks[i][j * block_size:(j + 1) * block_size,
                                k * block_size:(k + 1) * block_size]
                if (block_out_mask.sum() == 0) | (block_out_mask.sum() == block_size ** 2) | (block_gt_mask.sum() == 0):
                    continue
                    # 如果全为0，则全为背景部分，如果全为1，则是实例内部
                else:
                    # 不全为0，则为边缘，对应提取边缘所在的块
                    block_indexs.append(block_index)
                    block_out_masks.append(block_out_mask)
                    block_gt_masks.append(gt_masks[i][j * block_size:(j + 1) * block_size,
                                          k * block_size:(k + 1) * block_size])
                    block_edges.append(edges[i][j * block_size:(j + 1) * block_size,
                                       k * block_size:(k + 1) * block_size])

    print('ok')
    return block_indexs, block_out_masks, block_gt_masks, block_edges


def Check_splid(block_indexs, block_out_masks, block_gt_masks, block_edges, index):
    from matplotlib import pyplot as plt
    print(block_indexs[index])
    plt.imshow(edges[block_indexs[index][0]])
    plt.title("full edge of " + str(block_indexs[index][0]))
    plt.show()

    plt.imshow(block_edges[index])
    plt.title("edges of" + str(block_indexs[index]))
    plt.show()
    plt.imshow(block_out_masks[index])
    plt.title("out mask of" + str(block_indexs[index]))
    plt.show()
    plt.imshow(block_gt_masks[index])
    plt.title("gt mask of" + str(block_indexs[index]))
    plt.show()

