---
title: YOLOv3源码阅读：test_single_image.py
date: 2019-05-22 16:27:38
tags: [深度学习, 目标检测, YOLOv3]
categories: 目标检测
toc: true
mathjax:  true
thumbnail: gallery/DeepLearning.jpg
---

##### 一、YOLO简介  

&emsp;&emsp;YOLO（You Only Look Once）是一个高效的目标检测算法，属于One-Stage大家族，针对于Two-Stage目标检测算法普遍存在的运算速度慢的缺点，YOLO创造性的提出了One-Stage。也就是将物体分类和物体定位在一个步骤中完成。YOLO直接在输出层回归bounding box的位置和bounding box所属类别，从而实现one-stage。  

&emsp;&emsp;经过两次迭代，YOLO目前的最新版本为[YOLOv3](<https://pjreddie.com/media/files/papers/YOLOv3.pdf>)，在前两版的基础上，YOLOv3进行了一些比较细节的改动，效果有所提升。  

&emsp;&emsp;本问正是希望可以将源码加以注释，方便自己学习，同时也愿意分享出来和大家一起学习。由于本人还是一学生，如果有错还请大家不吝指出。  

<!--more-->

&emsp;&emsp;本文参考的源码地址为：<https://github.com/wizyoung/YOLOv3_TensorFlow>


##### 二、代码和注释  

&emsp;&emsp;文件目录：YOUR_PATH\\YOLOv3_TensorFlow-master\\test_single_image.py

&emsp;&emsp;需要注意的是，我们默认输入图片尺寸为$[416, 416]$。  

```python
# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3

# 设置命令行参数，具体可参见每一个命令行参数的含义
parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("input_image", type=str,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

# 处理anchors，这些anchors是通过数据聚类获得，一共9个，shape为：[9, 2]。
# 需要注意的是，最后一个维度的顺序是[width, height]
args.anchors = parse_anchors(args.anchor_path)

# 处理classes， 这里是将所有的class的名称提取了出来，组成了一个列表
args.classes = read_class_names(args.class_name_path)

# 类别的数目
args.num_class = len(args.classes)

# 根据类别的数目为每一个类别分配不同的颜色，以便展示
color_table = get_color_table(args.num_class)

# 读取图片
img_ori = cv2.imread(args.input_image)

# 获取图片的尺寸
height_ori, width_ori = img_ori.shape[:2]

# resize，根据之前设定的尺寸值进行resize，默认是[416, 416]，还是[width, height]的顺序
img = cv2.resize(img_ori, tuple(args.new_size))

# 对图片像素进行一定的数据处理
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

# TF会话
with tf.Session() as sess:
    # 输入的placeholder，用于输入图片
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    # 定义一个YOLOv3的类，在后面可以用来做模型建立以及loss计算等操作，参数分别是类别的数目和anchors
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        # 对图片进行正向传播，返回多张特征图
        pred_feature_maps = yolo_model.forward(input_data, False)
    # 对这些特征图进行处理，获得计算出的bounding box以及属于前景的概率已经每一个类别的概率分布
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    # 将两个概率值分别相乘就可以获得最终的概率值
    pred_scores = pred_confs * pred_probs

    # 对这些bounding boxes和概率值进行非最大抑制（NMS）就可以获得最后的bounding boxes和与其对应的概率值以及标签
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, nms_thresh=0.5)

    # Saver类，用以保存和恢复模型
    saver = tf.train.Saver()
    # 恢复模型参数
    saver.restore(sess, args.restore_path)

    # 运行graph，获得对应tensors的具体数值，这里是[boxes, scores, labels]，对应于NMS之后获得的结果
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    # 将坐标重新映射到原始图片上，因为前面的计算都是在resize之后的图片上进行的，所以需要进行映射
    boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
    boxes_[:, 3] *= (height_ori/float(args.new_size[1]))

    # 输出
    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    # 绘制并展示，保存最后的结果
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)

```





