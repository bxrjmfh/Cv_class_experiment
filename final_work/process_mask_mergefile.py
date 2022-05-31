import os.path
import pickle
from global_config import OUT_MASK_NP_PATH,OUT_MASK_NP
from os import listdir
from os.path import isfile, join
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

onlyfiles = [f for f in listdir(OUT_MASK_NP_PATH) if isfile(join(OUT_MASK_NP_PATH, f))]
onlyfiles.sort()

masks = []
for f in onlyfiles:
    logging.info("proceduring "+f)
    f_path=os.path.join(OUT_MASK_NP_PATH,f)
    with open(f_path, 'rb') as infile:
        masks+=pickle.load(infile)
    logging.info("successful")
    #   todo:排除某些字段     （id不满足，加载id）（mask数量大于1）（同时排除相关的图片）

with open(OUT_MASK_NP, 'wb') as outfile:
    pickle.dump(masks, outfile, pickle.HIGHEST_PROTOCOL)
logging.info("successful !")
