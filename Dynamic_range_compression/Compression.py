from PIL import Image
import numpy as np
from numpy import asarray
import math


def load_img_as_array(filepath:str):
    # "../cat.jpeg"
    pil_im = Image.open(filepath).convert('L')
    return asarray(pil_im)

def Compression(array,fileneme:str,c:int):
    max_list = []
    min_list =[]
    for i in range(array.shape[0]):
        max_list.append(max(array[i]))
        min_list.append(min(array[i]))
    max_val=max(max_list)
    min_val=min(min_list)
    a= []
    a.append([math.log(abs(max_val)+1),1])
    a.append([math.log(abs(min_val)+1),1])
    a=np.array(a)
    b=np.array([max_val,min_val])
    x=np.linalg.solve(a,b)
    print(a)
    print(b)
    print(x)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j]=x[0]*math.log(abs(array[i][j])+1)+x[1]

    im=Image.fromarray(array)
    im.save(fp=fileneme+".png",format="png")

Compression(load_img_as_array("../sky.png"), "sky_compression",80)