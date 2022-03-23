import time

from PIL import Image
from numpy import asarray
import numpy as np

def add_HE_fun():
    hist = [0] * (new_img.max() - new_img.min() + 1)
    for i in range(len(img)):
        for j in range(len(img[0])):
            agrey = new_img[i][j]
            hist[agrey + new_img.min()] += 1

    acc_hist = []
    acc_hist.append(float(hist[0] / len(hist)))
    for i in range(1, len(hist)):
        acc_hist.append(acc_hist[i - 1] + float(hist[i] / (len(img) * len(img[0]))))
        print(acc_hist)
    small_partition = new_img.min()
    print(small_partition)
    # time.sleep(5)
    for i in range(len(img)):
        for j in range(len(img[0])):
            agrey = new_img[i][j]
            new_grey = round(img[i][j] + min(acc_hist[agrey - small_partition] * 256, 255))
            print(new_grey)
            new_img[i][j] = new_grey

pil_im = Image.open("img.png").convert('L')
# pil_im.show()
# pil_im.thumbnail((pil_im.width*0.2,pil_im.height*0.2))
pil_im.save(fp="Penguins.png",bitmap_format="png")
im_array=asarray(pil_im)

img = im_array.astype('int32')
# 转换数组类型，避免溢出
new_img = img.copy()
for i in range(1, len(img) - 1):
    for j in range(1, len(img[0]) - 1):
        sum = int(int(- img[i - 1][j]) - img[i][j - 1] +4* img[i][j] - img[i][j + 1] - img[i + 1][j])
        new_img[i][j] = sum
        # img[i][j] = sum

# 对卷积图像进行灰度平移：
max_grey = new_img.max()
min_grey = new_img.min()
a=[]
a.append([max_grey,1])
a.append([min_grey,1])
a=np.array(a)
b=np.array([255,0])
x=np.linalg.solve(a,b)
print(x)
time.sleep(2)

for i in range(len(img)):
    for j in range(len(img[0])):
        new_img[i][j]=new_img[i][j]*x[0]+x[1]

average_val = new_img.mean()
for i in range(len(img)):
    for j in range(len(img[0])):
        new_img[i][j]=new_img[i][j]-average_val
        new_img[i][j] = round(min((img[i][j]-new_img[i][j]),255))


new_img = new_img.astype('uint8')
# 转化为uint8
im = Image.fromarray(new_img)
im.save(fp="/Data/Rice_Bowl/Python/CV/filtering/linear_penguins.png", format="png")
