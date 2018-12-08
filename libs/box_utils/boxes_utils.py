#coding:utf-8
from __future__ import absolute_import,print_function,division

import  tensorflow as tf

def clip_boxes_to_img_boundaries(decode_boxes,img_shape):
    """

    :param decode_boxes:
    :param img_shape:
    :return: decode boxes,and already clip to boundaries
    """

    with tf.name_scope('clip_boxes_to_img_boundaries'):
        ymin,xmin,ymax,xmax = tf.unstack(decode_boxes,axis=1)
        img_h,img_w = img_shape[1],img_shape[2]
        xmin = tf.maximum(xmin,0.0)
        xmin = tf.minimum(xmin,tf.cast(img_w,tf.float32))

        ymin = tf.maximum(ymin,0.0)
        ymim = tf.minimum(ymin,tf.cast(img_h,tf.float32)) #avoid xmin > img_w,ymin > img_h

        xmax = tf.minimum(xmax,tf.cast(img_w,tf.float32))
        ymax = tf.minimum(ymax,tf.cast(img_h,tf.float32))

        return tf.transpose(tf.stack([ymin,xmin,ymax,xmax]))


def filter_outside_boxes(boxes,img_w,img_h):
    """

    :param boxes: boxes with format [xmin,ymin,xmax,ymax]
    :param img_w: weight of img
    :param img_h: height of img
    :return: indices of anchors that not outside the image boundary
    """

    with tf.name_scope("filter_outside_boxes"):

        ymin,xmin,ymax,xmax = tf.unstack(boxes,axis=1)
        xmin_index = tf.greater_equal(xmin,0)
        ymin_index = tf.greater_equal(ymin,0)
        xmax_index = tf.less_equal(xmax,img_w)
        ymax_index = tf.less_equal(ymax,img_h)

        indices = tf.transpose(tf.stack([ymin_index,xmin_index,ymax_index,xmax_index]))
        indices = tf.cast(indices,dtype=tf.int32)
        indices = tf.reduce_sum(indices,axis=1)
        indices = tf.where(tf.equal(indices,tf.shape(boxes)[1]))

        return tf.reshape(indices,[-1,])

def nmx_boxes(decode_boxes,scores,iou_threshold,max_output_size,name):

    """
    1) NMS
    2) get maximum num of proposals

    :param decode_boxes:
    :param scores:
    :param iou_threshold:
    :param max_output_size:
    :param name:
    :return: valid indices
    """
    valid_indices = tf.image.non_max_suppression(boxes=decode_boxes,
                                                 scores=scores,
                                                 max_output_size=max_output_size,
                                                 iou_threshold=iou_threshold,
                                                 name=name)
    return valid_indices

def padd_boxes_with_zeros(boxes,scores,max_num_of_boxes):
    """
    num of boxes less than max num of boxes,so it need to pad with zeros[0,0,0,0]
    :param boxes:
    :param scores: [-1]
    :param max_num_of_boxes:
    :return:
    """


    pad_num = tf.cast(max_num_of_boxes,tf.int32) - tf.shape(boxes)[0]

    zero_boxes = tf.zeros(shape=[pad_num,4],dtype=boxes.dtype)
    zero_scores = tf.zeros(shape=[pad_num],dtype=scores.dtype)

    final_boxes = tf.concat([boxes,zero_boxes],axis=0)
    final_scores = tf.concat([scores,zero_scores],axis=0)

    return final_boxes,final_scores





