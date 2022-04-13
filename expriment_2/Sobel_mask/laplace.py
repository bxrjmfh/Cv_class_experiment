import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

from PIL import Image
from numpy import asarray
import numpy as np
import math

def mask_slid_for_3x3 (img,mask_x,mask_y,thershold=140,file_name='pic'):
    (x_length, y_length) = img.shape
    img_x = img.astype('int32')
    img_y = img.astype('int32')
    img_g_1 = img.astype('int32')
    img_g_2 = img.astype('int32')
    img_g_i = img.astype('int32')
    #   复制备份
    psubing_img = np.pad(img, ((1, 1), (1, 1)), 'edge').astype('int32')
    #     使用复制填充边缘
    for i in range(x_length):
        for j in range(y_length):
            block = psubing_img[i:i + 3, j:j + 3]
            temp_x = block * mask_x
            temp_y = block * mask_y
            temp_x = temp_x.sum()
            temp_y = temp_y.sum()
            img_x[i][j] = temp_x
            img_y[i][j] = temp_y
            img_g_1[i][j] = abs(temp_x)+abs(temp_y)
            img_g_2[i][j] = math.sqrt(temp_x**2+temp_y**2)
            img_g_i[i][j] = max(temp_y,temp_x)

    single_dir_pic = [img_x,img_y]
    for i,img_copy in enumerate(single_dir_pic):
        sub_img = img_copy - img
        sub_img -= sub_img.min()
        sub_img = sub_img.astype('float64')
        sub_img *= (255.0 / sub_img.max())
        sub_img = sub_img.astype('uint8')
        im = Image.fromarray(sub_img)
        im.save(fp='/Data/Rice_Bowl/Python/Cv_class_experiment-master/expriment_2/Laplace/' + file_name + str(i)+ '.png',
                format='png')

    grad_pic =[img_g_1,img_g_2,img_g_i]
    for i, img_copy in enumerate(grad_pic):
        sub_img = img_copy - img
        sub_img -= sub_img.min()
        sub_img = sub_img.astype('float64')
        sub_img *= (255.0 / sub_img.max())
        sub_img = sub_img.astype('uint8')
        sub_img_b = np.where(sub_img > thershold, 255, 0)
        im_b = Image.fromarray(sub_img_b.astype('uint8'))
        im_b.save(fp='/Data/Rice_Bowl/Python/Cv_class_experiment-master/expriment_2/Laplace/' + file_name+str(i+2) + 'b.png',
                  format='png')
    logging.info("finished !")

pil_im = Image.open("sky.png").convert('L')
im_array = asarray(pil_im)

Sobel_mask_x = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])

Sobel_mask_y = np.array([[1,2,1],
                         [0,0,0],
                         [-1,-2,-1]])


mask_slid_for_3x3(im_array,Sobel_mask_x,Sobel_mask_y,thershold=110,file_name='sky_sobel_110_')


