from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def k_largest_index_argpartition_v2(a, k):
    idx = np.argpartition(a.ravel(),a.size-k)[-k:]
    return np.column_stack(np.unravel_index(idx, a.shape))

def DoG_Corner_Dection(img=None,thershold = 5.0,sigma = 2**(0.5),num_DoG_images =4,m=1.5,
                       filename = 'obj',top_k_range = [5,10,20,30,40,80,160]
                        ):
    gaussian_images = []
    DoG_images = []
    gaussian_images.append(img)
    num_guassian_images=num_DoG_images+1

    # 根据论文中的结论，sigma的大小由噪声情况确定。
    # 添加高斯模糊的图片
    dir_name = "DoG_"+filename + '_sigma_' + str(int(sigma)) + '_T_' + str(thershold)+"_m_"+str(m)
    os.mkdir(dir_name)
    os.mkdir(dir_name+'/DoG_pic')
    for i in range(num_guassian_images):
        r = sigma*(i+1)*m
        img_temp = pil_im.filter(ImageFilter.GaussianBlur(radius=r))
        plt.imshow(np.asarray(img_temp),cmap=cm.gray)
        name_temp = str(i)+"_sigma_"+str(int(r))
        plt.title(name_temp)
        plt.savefig(dir_name+'/DoG_pic/'+name_temp)
        gaussian_images.append(np.asarray(img_temp).astype('int32'))

    # create DoG
    # find the key point
    for k in top_k_range:
        key_points = []
        for i in range(num_DoG_images):
            DoG_images.append(abs(gaussian_images[i]-gaussian_images[i+1]))
            temp_key_point_list = k_largest_index_argpartition_v2(DoG_images[i], k)
            for j,key in enumerate(temp_key_point_list):
                if DoG_images[i][key[0]][key[1]]<thershold:
                    temp_key_point_list[j]=[0,0]
            key_points.append(temp_key_point_list)

        key_points = np.asarray(key_points)
        key_points = np.unique(key_points,axis=0)
        key_points = key_points.T

        x = key_points[0].tolist()
        y = key_points[1].tolist()
        plt.figure()
        plt.scatter(y, x, color='r')
        plt.imshow(img, cmap=cm.gray)
        fig_name = str(k)
        plt.title(fig_name)
        plt.savefig(dir_name+'/'+fig_name)
        plt.show()

pil_im = Image.open('obj_8bit.png').convert('L')
img = np.asarray(pil_im)
DoG_Corner_Dection(img=img)


