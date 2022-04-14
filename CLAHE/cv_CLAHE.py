import numpy as np
import cv2

img = cv2.imread('regular_people.png',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imwrite('CLAHE_people.png',cl1)