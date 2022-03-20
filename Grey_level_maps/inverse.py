from PIL import Image
from numpy import asarray
import numpy

def load_img_as_array(filepath:str):
    # "../cat.jpeg"
    pil_im = Image.open(filepath).convert('L')
    return asarray(pil_im)

def inverse(array,fileneme:str):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j]=255-array[i][j]

    im=Image.fromarray(array)
    im.save(fp=fileneme+".png",format="png")

inverse(load_img_as_array("../cat.jpeg"), "cat_inverse")
