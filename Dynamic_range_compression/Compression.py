from PIL import Image
from numpy import asarray
import math


def load_img_as_array(filepath:str):
    # "../cat.jpeg"
    pil_im = Image.open(filepath).convert('L')
    return asarray(pil_im)

def Compression(array,fileneme:str,c:int):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j]=c*math.log(abs(array[i][j])+1,10)

    im=Image.fromarray(array)
    im.save(fp=fileneme+".png",format="png")

Compression(load_img_as_array("../sky.png"), "sky_compression",80)