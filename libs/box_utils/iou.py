#coding:utf-8
from __future__ import absolute_import,print_function,division
import tensorflow as tf

def iou_calculate(boxes_1,boxes_2):
    """

    :param boxes_1:[N,4] [ymin,xmin,ymax,xmax]
    :param boxes_2: [N,4] [ymin,xmin,ymax,xmax]
    :return:
    """

    with tf.name_scope('iou_calculate'):
        ymin_1,xmin_1,ymax_1,xmax_1 = tf.split(boxes_1,4,axis=1) # ymin_1 shape is [N,1]
        ymin_2,xmin_2,ymax_2,xmax_2 = tf.split(boxes_2,4,axis=1)# ymin_2 shape is [N,1]

        x_min = tf.maximum(xmin_1,xmin_2)
        x_max = tf.minimum(xmax_1,xmax_2)
        y_min = tf.maximum(ymin_1,ymin_2)
        y_max = tf.minimum(ymax_1,ymax_2)
        overlap_w = tf.maximum(0,x_max-x_min)
        overlap_h = tf.maximum(0,y_max - y_min)

        overlap = overlap_w * overlap_h

        area1 = (xmax_1 - xmin_1) * (ymax_1 - ymax_2)
        area2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_1)
        iou = overlap/(area1 + area2 - overlap)

        return iou