from PIL import Image
from numpy import asarray
pil_im = Image.open("../cat.jpeg").convert('L')
# pil_im.show()
pil_im.save(fp="cat.png",bitmap_format="png")
im_array=asarray(pil_im)

