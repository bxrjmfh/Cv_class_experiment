import cv2 as cv
import numpy as np
import pickle
from global_config import IMAGE_DIR_NP,EDGE_DIR_NP
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

with open(IMAGE_DIR_NP, 'rb') as infile:
    images = pickle.load(infile)

edges = []
i = 1
size = len(images)
for img in images:
    logging.info("processing {}/{}".format(i,size))
    i+=1
    blurred_img = cv.blur(img, ksize=(5, 5))
    med_val = np.median(img)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    edge = cv.Canny(image=img, threshold1=lower, threshold2=upper)
    # plt.imshow(edge, cmap='gray')
    # plt.show()
    edges.append(edge)

with open(EDGE_DIR_NP,"wb") as outfile:
    pickle.dump(edges, outfile, pickle.HIGHEST_PROTOCOL)


