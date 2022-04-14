from PIL import Image
from numpy import asarray
pil_im = Image.open("../sky.jpg").convert('L')
# pil_im.show()
pil_im.thumbnail((pil_im.width*0.2,pil_im.height*0.2))
pil_im.save(fp="../sky.png",bitmap_format="png")
im_array=asarray(pil_im)

