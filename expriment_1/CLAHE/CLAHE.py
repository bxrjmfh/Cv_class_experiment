import time
from PIL import Image
from numpy import asarray
import numpy as np
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

def process_pic():
    img = Image.open('people.png').convert('L')
    img = img.resize((650, 500), Image.ANTIALIAS)
    img.save('regular_people.png')


def EQ(img):
    leng=img.max()-img.min()+1
    hist = [0] * leng
    w = len(img[0])
    h = len(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            grey_val = img[i][j]
            hist[grey_val-img.min()] += 1.0
    hist[0] = hist[0] / w / h
    acc_hist = []
    acc_hist.append(hist[0])
    for i in range(1, leng):
        hist[i] = hist[i] / w / h
        acc_hist.append(acc_hist[i - 1] + hist[i])
        logging.info(acc_hist)
    for i in range(len(img)):
        for j in range(len(img[0])):
            logging.info('{} / {}'.format(i,j))
            grey_val = img[i][j]
            if grey_val<img.min():
                time.sleep(10)
            # logging.info('grey_val,img.min(),len(acc_hist) {},{},{}'.format(grey_val,img.min(),len(acc_hist)))
            temp_val =leng* acc_hist[min(grey_val-img.min(),len(acc_hist)-1)]

            img[i][j] = round(max( temp_val, 0))
    print(img)
    # time.sleep(1)
    return img

img = Image.open('regular_people.png')
imga = asarray(img)
THRESHOLD = 40

for i in range(50):
    for j in range(65):
        #         process one block

        logging.info("Mapping step {} / {}".format(i * 13 + j,130))
        block_size =10
        block = imga[i * block_size:(i + 1) * block_size,j * block_size:(j + 1) * block_size]

        min_val = block.min()
        acc = 0
        # accumulation of over value
        max_val = min_val + THRESHOLD
        for k in range(block_size):
            for l in range(block_size):
                if block[k][l] > THRESHOLD:
                    acc = block[k][l] - THRESHOLD
                    block[k][l] = THRESHOLD
        logging.info(acc)
        # time.sleep(0.1)
        block + round(acc / (block_size * block_size))
        EQ(block)
        imga[i * block_size:(i + 1) * block_size,j * block_size:(j + 1) * block_size] = block

img = Image.fromarray(imga)
img.save('CLAHE_people.png','png')
