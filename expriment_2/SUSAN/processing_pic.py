from PIL import Image
from numpy import asarray
pil_im = Image.open("obj.png").convert('L')
# 转化为灰度图像
pil_im.save(fp='obj_8bit.png',bitmap_format='png')
