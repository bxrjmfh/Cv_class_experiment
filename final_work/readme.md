# 基于mask-rcnn 的图像掩模改进

## 背景调查：

## 基本思路：
考虑到mask-rcnn的实例分割创建的mask对于实例的覆盖效果不佳，本文尝试通过提取图像边缘信息，结合mask-rcnn的实例预测结果，通过多输入卷积神经网络处理输出结果。模型输入为裁剪后的图像边缘图块，以及对应的边缘信息（例如 paper/block_positive）的图片。

## 模型架构：
图片：md文件下的model.png  ，将原始的mask和边缘信息都输入卷积层后相加合并，再通过一系列卷积层进行处理，在最后通过两个密集连接层进行处理后输出最后结果。

## 训练数据加载和处理：
本模型训练数据取自于coco数据集，通过final_work/coco/coco_load_test.py 加载coco数据集，选择只属于一类的存储图像集合以及模板，便于训练模型
输入1 为 mask-rcnn 输出的 mask ，通过文件 load_mask_rcnn 进行处理。得到mask后，对于统一类别的实例进行组合，如图（merge_mask_out）中所示
输入2 为原始图像的边缘响应信息，通过edgedetector文件中的canny算子处理后获得边缘信息，
加载数据结果在loadresult中显示，其中gt mask 为数据集原始标注的信息，out——mask为mask-rcnn输出的mask信息。

得到训练数据后，去除类别不一致的输入信息（wash——data），进行下一步处理

##  训练模型：
通过preprocess——toblock 提取图像边缘所在的图块，图块大小为block——size，最后在train——model 中进行训练。