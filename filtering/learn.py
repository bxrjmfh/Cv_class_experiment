from PIL import Image
from numpy import asarray
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

hist = [0]*(new_img.max()-new_img.min()+1)
for i in range( len(img) ):
    for j in range( len(img[0]) ):
        agrey=new_img[i][j]
        hist[agrey+new_img.min()]+=1

acc_hist = []
acc_hist.append(float(hist[0]/len(hist)))
for i in range(1,len(hist)):
    acc_hist.append(acc_hist[i-1]+float(hist[i]/len(hist)))
small_partition = new_img.min()

for i in range(len(img)):
    for j in range(len(img[0])):
        agrey = new_img[i][j]
        new_img[i][j] = round(max(acc_hist[agrey-small_partition]*256,255))
img = img.astype('uint8')
# 转化为uint8
im = Image.fromarray(img)
im.save(fp="linear_penguins.png", format="png")
