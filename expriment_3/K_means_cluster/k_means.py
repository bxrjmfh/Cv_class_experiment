import datetime
import time

from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import random
import os
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")
ROOT_PATH = "/Data/Rice_Bowl/Python/Cv_class_experiment-master/expriment_3/K_means_cluster"

def Show_raw_pic(img=None,name='nope'):
    # 尝试显示不同通道的图片
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # 添加三维子图
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    flatten_img = img.reshape((img.shape[0] * img.shape[1], 3))
    ax.scatter(flatten_img[:, 0], flatten_img[:, 1], flatten_img[:, 2], s=0.5)
    plt.title('raw_RGB_mapping')
    plt.savefig(name + '_raw_RGB_mapping')
    plt.show()

# kmeans_test
def Get_distances(x,centers):
    return ((centers-x)**2).sum(axis=1)
#     向高维度扩展，随后向减乘方，并在通道轴上求和

def Get_distances_3D(x,centers):
    return ((centers-x)**2).sum(axis=2)
#     向高维度扩展，随后向减乘方，并在通道轴上求和

def Get_closest_index(dist_s):
    return np.argmin(dist_s,axis=0)
# 返回索引最小的数值

def Get_closest_index_2D(dist_s_2D):
    return np.argmin(dist_s_2D, axis=1)
    # 返回索引最小的数值,此时索引值为1

def Draw_res(img,name,k_value):
    # 返回原有图像
    dirname = name+'/'+str(k_value)
    try:
        os.mkdir(name)
    except Exception as e:
        print(e)
    try:
        os.mkdir(dirname)
    except Exception as e:
        print(e)

    plt.imshow(img)
    plt.title(name+str(k_value))
    plt.savefig(dirname+'/'+name+'_'+str(k_value)+'_'+datetime.datetime.now().strftime('%H_%M_%S'))
    plt.show()
    plt.clf()

def Draw_SSE(SSE,name,k_value):
    dirname = name + '/' + str(k_value)
    plt.plot(range(len(SSE)),SSE)
    plt.scatter(range(len(SSE)),SSE,c='red',s=20)
    plt.title('SSE of '+name+str(k_value)+'_'+datetime.datetime.now().strftime('%H_%M_%S'))
    plt.savefig(dirname + '/SSE'+datetime.datetime.now().strftime('%H_%M_%S'))
    plt.show()
    plt.clf()
def Old_k_means(img=None,k = 4,iter_num = 50,threshold = 0.1,name=''):
    # threshold 设置停止迭代阈值：
    img_copy = np.copy(img)
    flatten_img = img_copy.reshape((img_copy.shape[0] * img_copy.shape[1], img_copy.shape[2]))
    # np.random.shuffle(flatten_img)
    # 初始化种子
    random.seed(0)
    # 初始化核心列表
    centers = []
    # SSE记录
    SSE = []

    ini_centers_index = random.sample(range(flatten_img.shape[0]), k)
    for i in range(k):
        centers.append(flatten_img[ini_centers_index[i]])
    # 结果
    results = np.asarray([0] * len(flatten_img))

    for i in range(iter_num):
        #     每次迭代
        tic = time.time()
        sse = 0
        for j in range(len(flatten_img)):
            #         遍历每一个点
            x = flatten_img[j]
            dist_s = Get_distances(x, centers)
            # 计算到各中心的距离
            min_index = Get_closest_index(dist_s)
            results[j] = min_index
            # 记录聚类结果
            sse += dist_s[min_index]
            # 记录SSE数值

        for kk in range(len(centers)):
            cata = flatten_img[results == kk]
            # 取一个类别中的所有点
            new_center = cata.mean(axis=0)
            # 计算新的中心位置
            centers[kk] = new_center
            # 更新中心位置
        toc = time.time()
        SSE.append(sse / len(flatten_img))

        if i>=1:
            delta_SSE = SSE[len(SSE) - 2] - SSE[len(SSE) - 1]
        else:
            delta_SSE =99999
        print()
        logging.info("processing the " + str(i) + ' th iteration ...')
        logging.info('Total time is {:.3f}'.format((toc - tic) / 50))
        logging.info('The centers is ' + str(centers))
        logging.info("SSE : {:.3f}".format(sse / len(flatten_img)))
        logging.info('ΔSSE : {:.3f}'.format(delta_SSE))
        print()
        if abs(delta_SSE) < threshold:
            break

    for kk in range(k):
        flatten_img[results==kk]=centers[kk]
        # 将对应类别赋予中心值

    Draw_res(flatten_img.reshape(img.shape),name,k)
    Draw_SSE(SSE,name,k)

def New_k_means(img=None,k = 4,iter_num = 50,threshold = 0.1,name=''):
    # threshold 设置停止迭代阈值：
    img_copy = np.copy(img)
    flatten_img = img_copy.reshape((img_copy.shape[0] * img_copy.shape[1], img_copy.shape[2]))
    # 初始化种子
    random.seed(0)
    # 初始化核心列表
    centers = []
    # SSE记录
    SSE = []

    ini_centers_index = random.sample(range(flatten_img.shape[0]), k)
    for i in range(k):
        centers.append(flatten_img[ini_centers_index[i]])
    # 结果
    centers = np.asarray(centers).astype('float32')
    results = np.asarray([0] * len(flatten_img))

    for i in range(iter_num):
        #     每次迭代
        tic = time.time()
        sse = 0

        dist_2D = Get_distances_3D(flatten_img[:,None,:],centers[None,:,:])
        # 以ndarray形式计算，使数据在不存在的那个轴上扩展，得到最后的结果
        results = Get_closest_index_2D(dist_2D)

        for kk in range(len(centers)):
            sse += np.sum(dist_2D[results==kk][kk])
            cata = flatten_img[results == kk]
            # 取一个类别中的所有点
            new_center = cata.mean(axis=0)
            # 计算新的中心位置
            centers[kk] = new_center
            # 更新中心位置
        toc = time.time()
        SSE.append(sse / len(flatten_img))

        if i>=1:
            delta_SSE = SSE[len(SSE) - 2] - SSE[len(SSE) - 1]
        else:
            delta_SSE =99999
        print()
        logging.info("processing the " + str(i) + ' th iteration ...')
        logging.info('Total time is {:.3f}'.format((toc - tic) / 50))
        logging.info('The centers is ' + str(centers))
        logging.info("SSE : {:.3f}".format(sse / len(flatten_img)))
        logging.info('ΔSSE : {:.3f}'.format(delta_SSE))
        print()
        if abs(delta_SSE) < threshold and i>=8:
            break

    for kk in range(k):
        flatten_img[results==kk]=centers[kk]
        # 将对应类别赋予中心值

    Draw_res(flatten_img.reshape(img.shape),name,k)
    Draw_SSE(SSE,name,k)


pil_im = Image.open('/Data/Rice_Bowl/Python/Cv_class_experiment-master/expriment_3/K_means_cluster/daqidawei.jpg')
# 将彩色图片转化为16 bit的 RGB 格式
# pil_im.thumbnail((400,300))
# 调整图像大小
img = asarray(pil_im)
name='daqidawei_full_image'
# New_k_means(img=img,name=name,k=5)
# New_k_means(img=img,name=name,k=10)
# New_k_means(img=img,name=name,k=15)
# New_k_means(img=img,name=name,k=20)
New_k_means(img=img,name=name,k=40)







