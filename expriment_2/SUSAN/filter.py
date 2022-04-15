import numpy as np
from PIL import Image
from numpy import asarray


def averange(img):
    copy_img = np.copy(img)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            sum=int(img[i-1][j-1])+img[i-1][j]+img[i-1][j+1]+img[i][j-1]+img[i][j]+img[i][j+1]+img[i+1][j-1]+img[i+1][j]+img[i+1][j+1]
            aver=sum/9
            copy_img[i][j]=aver
    im = Image.fromarray(copy_img)
    im.save(fp="aver_obj.png", format="png")

pil_im = Image.open('obj.png').convert('L')
img = asarray(pil_im)
averange(img)