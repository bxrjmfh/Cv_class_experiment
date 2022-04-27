from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")
ROOT_PATH = "/Data/Rice_Bowl/Python/Cv_class_experiment-master/expriment_3/Thershold_Partition"

def Get_Partition(img):
    grey_value, grey_count = np.unique(img, return_counts=True)
    # 获得图像的灰度值，以及对应的计数
    grey_map = {grey_value[i]: grey_count[i] for i in range(len(grey_value))}
    # 创建灰度映射的字典
    histgram = []
    # 创建灰度直方图
    for i in range(256):
        try:
            histgram.append(grey_map[i])
        #     如果字典中的项存在，则添加到直方图中
        except:
            histgram.append(0)
        #     不存在那么该灰度为0
    plt.bar(range(len(histgram)), histgram)
    # 展示灰度图

    z1 = np.polyfit(range(len(histgram)), histgram, 40)
    p1 = np.poly1d(z1)
    # 多项式子拟合
    test_val = p1(range(len(histgram)))
    # 存储多项式拟合结果
    a = np.polyder(p1)
    der_fun = np.poly1d(a)
    # 求导数的根
    roots = np.roots(der_fun)
    real_roots = [int(n) for n in roots if (n.imag == 0) & (n.real < 256) & (n.real > 0)]
    # 在根中筛选二阶导数大于0的点
    der_2_fun = np.poly1d(np.polyder(der_fun))
    min_points = [n for n in real_roots if der_2_fun(n) > 0]

    plt.clf()
    plt.plot(range(len(histgram)), test_val, '#DE9717', label="fitting line")
    plt.bar(range(len(histgram)), histgram, color='#1F77B4', label='histgram')
    plt.scatter(min_points, p1(min_points).tolist(), s=[15] * len(min_points), color='r', label='partition point')
    plt.legend()
    plt.title("histgram and fitting result of " + str(len(p1)))
    plt.savefig("fitting_" + str(len(p1)))
    # plt.show()
    # 显示灰度分布图像以及极小值点（划分）
    plt.clf()
    return min_points

def Threshold_Partion(img = None,thresholds = [40,88,120,210],fname = 'stone'):
    thresholds.sort()
    # 整理阈值（可能多个）
    for i, T_1 in enumerate(thresholds):
        # 遍历所有阈值数组
        try:
            T_2 = thresholds[i + 1]
            # 对于有多个阈值的情况下，取出前一个阈值（划分为段）
            img = np.where(((T_1 < img) & (img < T_2)), ((T_1) + (T_2)) / 2, img)
            #  对于在T1T2阈值范围内的元素记录为两阈值的中间值。
        except:
            if i == 0:
                # 仅有一个阈值的情况
                img = np.where(img < T_1, T_1 / 2, (T_1 + 255) / 2)
            else:
                # 多个阈值且前边阈值段已经划分完成
                img = np.where(img > T_1, (T_1 + 255) / 2, img)

    name_str = fname + '_' + str(thresholds)
    plt.imshow(img, cmap=cm.gray)
    plt.title(name_str)
    plt.savefig(name_str)
    plt.show()
    plt.clf()



pil_im = Image.open('stone.png').convert('L')
img = asarray(pil_im)
Partition = Get_Partition(img)
Threshold_Partion(img,Partition,'stone')

# test grey map
