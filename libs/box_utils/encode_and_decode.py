#coding:utf-8
from __future__ import absolute_import,print_function,division
import tensorflow as tf

import numpy as np

def decode_boxes(encode_boxes,reference_boxes,scale_factors = None,name = 'decode'):

    """

    :param encode_boxes: [N,4]
    :param reference_boxes: [N,4]
    :param scale_factors: use for scale
    in the first stage,reference_boxes are anchors
    in the second stage,reference_boxes are proposals(decode) produces by rpn stage
    :param name:
    :return: decode boxes [N,4]
    """
    with tf.name_scope(name):
        t_xcenter,t_ycenter,t_w,t_h = tf.unstack(encode_boxes,axis=1)
        if scale_factors:
            t_xcenter /= scale_factors[0]
            t_ycenter /= scale_factors[1]
            t_w /= scale_factors[2]
            t_h /= scale_factors[3]

        reference_ymin,reference_xmin,reference_ymax,reference_xmax = tf.unstack(reference_boxes,axis=1)

        reference_x_center = (reference_xmax + reference_xmin) / 2.
        reference_y_center = (reference_ymax + reference_ymin) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin

        predict_xcenter = t_xcenter * reference_w + reference_x_center
        predict_ycenter = t_ycenter * reference_h + reference_y_center
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h

        predict_xmin = predict_xcenter - predict_w /2.
        predict_xmax = predict_xcenter + predict_w /2.
        predict_ymin = predict_ycenter - predict_h /2.
        predict_ymax = predict_ycenter + predict_h /2.

        return tf.transpose(tf.stack([predict_ymin,predict_xmin,predict_ymax,predict_xmax]))


def encode_boxes(unencode_boxes,reference_boxes,scale_factors = None,name = 'encode'):
    """

    :param unencode_boxes: [batch_size *H*W*num_anchors_per_location,4]
    :param reference_boxes: [H*W*num_anchors_per_location,4]
    :param scale_factors:
    :param name:
    :return: encode boxes [-1,4]
    """


    with tf.name_scope(name):
        ymin,xmin,ymax,xmax = tf.unstack(unencode_boxes,axis=1)

        reference_ymin,reference_xmin,reference_ymax,reference_xmax = tf.unstack(reference_boxes,axis=1)

        x_center = (xmin + xmax) /2.
        y_center = (ymin + ymax) / 2.
        w = xmax - xmin
        h = ymax - ymin

        reference_xcenter = (reference_xmin + reference_xmax) /2.
        reference_ycenter = (reference_ymin + reference_ymax) /2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin

        reference_w +=1e-8
        reference_h += 1e-8
        t_xcenter = (x_center - reference_xcenter) / reference_w
        t_ycenter = (y_center - reference_ycenter) / reference_h
        t_w = tf.log(w/reference_w)
        t_h = tf.log(h/reference_h)

        if scale_factors:
            t_xcenter *= scale_factors[0]
            t_ycenter *= scale_factors[1]
            t_w *= scale_factors[2]
            t_h *= scale_factors[3]

        return tf.transpose(tf.stack([t_xcenter,t_ycenter,t_w,t_h]))

