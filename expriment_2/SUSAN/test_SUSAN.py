from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import string
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

def k_largest_index_argpartition_v2(a, k):
    idx = np.argpartition(a.ravel(),a.size-k)[-k:]
    return np.column_stack(np.unravel_index(idx, a.shape))

radius = 10
img=None
threshold_T = 10
Adjust_G=3
Under_bound_R = 6
top_k_range = [5,10,20,30,40,80,160,320]
filename ='obj'
pil_im = Image.open('obj.png').convert('L')
img = asarray(pil_im)

(x_length, y_length) = img.shape
copy_img = img.astype('int32')
padding_img = np.pad(copy_img, ((radius, radius), (radius, radius)), 'edge')
# 扩展半径边缘的操作
x = np.arange(0, 2 * radius + 1)
y = np.arange(0, 2 * radius + 1)
circle_mask = (x[np.newaxis, :] - radius) ** 2 + (y[:, np.newaxis] - radius) ** 2 <= radius ** 2
# 创建可复用的模板
for i in range(x_length):
    for j in range(y_length):
        logging.info("processing : " + str((i, j)))
        block = np.copy(padding_img[i:i + 2 * radius + 1, j:j + 2 * radius + 1])
        #  裁剪处理的块
        center_expand_block = np.full(block.shape, copy_img[i][j])
        #  每个中心的块
        block -= center_expand_block
        block = abs(block)
        block_C = np.where(block > threshold_T, 0, 1)
        #  判断是否为和同值区域
        block = block_C.astype('bool')
        block *= circle_mask
        #  掩盖出圆形区域
        copy_img[i][j] = block.sum()
        #  得到每一个像素的核同值区域

threshold_G = int(Adjust_G * (circle_mask.sum() - 1) / 4)
# 指定几何阈值
img_R = np.where((threshold_G > copy_img) & (copy_img>Under_bound_R*radius), threshold_G - copy_img, 0)
# 输出响应
dir_name = filename + '_R_' + str(radius) + '_T_' + str(threshold_T) + '_ADJ_G_' + str(Adjust_G)\
    +'_Ud_'+str(Under_bound_R)
try:
    os.mkdir(dir_name)
    os.mkdir(dir_name+'/edge')
except Exception as e:
    print(e)
for top_k in top_k_range:
    coner_list = k_largest_index_argpartition_v2(img_R, top_k)
    coner_list_xy = coner_list.T
    x = coner_list_xy[0].tolist()
    y = coner_list_xy[1].tolist()
    plt.figure()
    plt.scatter(y, x, color='r')
    name_str =dir_name+'_' +str(top_k)
    if '.' in name_str:
        name_str = name_str.replace('.','_')
    plt.imshow(img, cmap=cm.gray)
    plt.title(name_str)
    plt.savefig(dir_name+'/'+name_str)
    plt.show()
    plt.close()

    for x_ ,y_ in zip(x,y):
        plt.figure()
        temp_block =np.copy(padding_img[x_:x_ + 2 * radius + 1, y_:y_ + 2 * radius + 1])
        plt.imshow(temp_block)
        plt.scatter(radius, radius, color='r')
        plt.title(str((x_, y_))+'_'+str(img_R[x_][y_]))
        plt.savefig(dir_name + '/edge/' + str((x_, y_))+'_'+str(img_R[x_][y_]))
        # plt.show()
        plt.close()


# SUSAN(img=img)
# SUSAN(radius=7,img=img)
# SUSAN(radius=9,img=img)
# SUSAN(radius=11,img=img)
# SUSAN(radius=9,img=img,threshold_T=20)
# SUSAN(radius=9,img=img,threshold_T=15)
# SUSAN(radius=9,img=img,threshold_T=10)
# SUSAN(radius=9,img=img,threshold_T=5)
# SUSAN(radius=11,img=img,threshold_T=5)
# SUSAN(filename='happy',radius=5,img=img,threshold_T=5,Adjust_G=3)
# SUSAN(filename='cha',radius=5,img=img,threshold_T=40,Adjust_G=1)





