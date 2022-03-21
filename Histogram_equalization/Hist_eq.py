import time

import PIL
import numpy as np
from PIL import Image
from numpy import asarray
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

def load_img_as_array(filepath:str):
    # "../cat.jpeg"
    pil_im = Image.open(filepath).convert('L')
    return asarray(pil_im)

def return_histogram_and_accumulation(img):
    hist=[0]*256
    w = len(img[0])
    h = len(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            grey_val=img[i][j]
            hist[grey_val]+=1.0
            logging.info("Mapping step {} / {}".format(i * len(img) + j,w*h))

    logging.info('out:'+str(hist))

    hist[0]=hist[0]/w/h
    acc_hist=[]
    acc_hist.append(hist[0])
    for i in range(1,256):
        hist[i]=hist[i]/w/h
        acc_hist.append(acc_hist[i-1]+hist[i])

    logging.info('out:' + str(acc_hist))
    # time.sleep(9000)
    return hist ,acc_hist

def mapping(img,filename):
    w=len(img[0])
    h=len(img)
    hist,acc_hist =return_histogram_and_accumulation(img)

    for i in range(len(img)):
        for j in range(len(img[0])):
            grey_val=img[i][j]
            img[i][j]=round(max(255.0*acc_hist[grey_val],0))
            logging.info("Save step {} / {}".format(i * len(img) + j,w*h))

    new_hist,new_acc_hist =return_histogram_and_accumulation(img)
    im = Image.fromarray(img)
    im.show()
    im.save(fp=filename + ".png", format="png")
    return img,hist,acc_hist,new_hist,new_acc_hist

img,hist,acc_hist,new_hist,new_acc_hist=mapping(load_img_as_array('../Dynamic_range_compression/sky_compression.png'),'hist_sky')
import matplotlib.pyplot as plt

plt.bar(range(len(hist)),hist)
plt.title("Hist of old pic")
plt.show()
plt.savefig('ho')
plt.bar(range(len(acc_hist)),acc_hist)
plt.title("Acc hist of old pic")
plt.show()
plt.savefig('aho')

plt.bar(range(len(new_hist)),new_hist)
plt.title("new_Hist of old pic")
plt.show()
plt.savefig('hn')

plt.bar(range(len(new_acc_hist)),new_acc_hist)
plt.title("new_Acc hist of old pic")
plt.show()
plt.savefig('ahn')

# Image.show()