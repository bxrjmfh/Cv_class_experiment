
import logging
import pickle
import matplotlib.pyplot as plt
import numpy as np

from global_config import IMAGE_DIR_NP,OUT_MASK_NP,EDGE_DIR_NP,ID_NP,GT_MASK_NP
from global_config import IMAGE_DIR_NP_U,OUT_MASK_NP_U,EDGE_DIR_NP_U,ID_NP_U,GT_MASK_NP_U
import logging
format ="%(asctime)s.%(msecs)05d : %(message)s"
logging.basicConfig(format=format,level=logging.INFO,datefmt="%H:%M:%S")

def Chose_Merge_or_Max(out_mask,true_id):
    ids = out_mask["class_ids"]
    main_mask = []
    for i,id in enumerate(ids):
        if id == true_id:
            main_mask.append(i)
            # plt.imshow(out_mask["masks"][:,:,i])
            # plt.show()

    if main_mask:
        union_mask = [out_mask["masks"][:,:,j] for j in main_mask]
        union_mask = np.asarray(union_mask)
        union_mask=union_mask.sum(axis= 0)
        union_mask = union_mask.astype('bool')
        # plt.imshow(union_mask)
        # plt.show()
        return union_mask,True
    else:
        return None,False

with open(IMAGE_DIR_NP, 'rb') as infile:
    images = pickle.load(infile)
with open(OUT_MASK_NP, 'rb') as infile:
    out_masks = pickle.load(infile)
with open(ID_NP, 'rb') as infile:
    ids = pickle.load(infile)
with open(GT_MASK_NP, 'rb') as infile:
    gt_masks = pickle.load(infile)
with open(EDGE_DIR_NP, 'rb') as infile:
    edges = pickle.load(infile)

images_u=[]
out_masks_u=[]
ids_u=[]
edges_u=[]
gt_masks_u=[]
# 创建筛选后的列表
for i,out_mask in enumerate(out_masks):
    union_mask,keep_flag=Chose_Merge_or_Max(out_mask,ids[i][0])
    logging.info("processing {:02d} th pic".format(i))
    if keep_flag:
        images_u.append(images[i])
        edges_u.append(edges[i])
        gt_masks_u.append(gt_masks[i][:,:,0])
        ids_u.append(ids[i][0])
        out_masks_u.append(union_mask)

with open(EDGE_DIR_NP_U, 'wb') as outfile:
    pickle.dump(edges_u, outfile, pickle.HIGHEST_PROTOCOL)
with open(GT_MASK_NP_U, 'wb') as outfile:
    pickle.dump(gt_masks_u, outfile, pickle.HIGHEST_PROTOCOL)
with open(ID_NP_U, 'wb') as outfile:
    pickle.dump(ids_u, outfile, pickle.HIGHEST_PROTOCOL)
with open(IMAGE_DIR_NP_U, 'wb') as outfile:
    pickle.dump(images_u, outfile, pickle.HIGHEST_PROTOCOL)
with open(OUT_MASK_NP_U, 'wb') as outfile:
    pickle.dump(out_masks_u, outfile, pickle.HIGHEST_PROTOCOL)

logging.info('finished')

def examination(i):
    plt.imshow(images_u[i])
    plt.title('images {:03d}'.format(i))
    plt.show()

    plt.imshow(gt_masks_u[i])
    plt.title('gt mask {:03d}'.format(i))
    plt.show()

    plt.imshow(out_masks_u[i])
    plt.title('out mask {:03d}'.format(i))
    plt.show()

    plt.imshow(edges_u[i])
    plt.title('edges {:03d}'.format(i))
    plt.show()
# plt.imshow(images[8])
# plt.show()
# Chose_Merge_or_Max(out_masks[8],None,ids[8])






