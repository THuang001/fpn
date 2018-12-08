#coding:utf-8
from __future__ import absolute_import,print_function,division
import tensorflow as tf

def enum_scales(base_anchor,anchor_scales,name = 'enum_scales'):

    """

    :param base_anchor: [y_center,x_center,w,h]
    :param anchor_scales: different scales ,like [0.5,1.,2.0]
    :param name:
    :return:return base anchors in different scales.
    Example:[[0,0,128,128],[0,0,256,256],[0,0,512,512]]
    """

    with tf.name_scope(name):
        anchor_scales = tf.reshape(anchor_scales,[-1,1])

        return base_anchor * anchor_scales
def enum_ratios(anchors,anchor_ratios,name = 'enum_ratios'):

    """

    :param anchors: base anchor in different scales
    :param anchor_ratios: ratio = h/w
    :param name:
    :return: base anchor in different scales and ratios
    """

    with tf.name_scope(name):
        _,_,hs,ws = tf.unstack(anchors,axis=1)
        sqrt_ratios = tf.sqrt(anchor_ratios)
        sqrt_ratios = tf.expand_dims(sqrt_ratios,axis=1)
        ws = tf.reshape(ws / sqrt_ratios,[-1])
        hs = tf.reshape(hs * sqrt_ratios,[-1])

        num_anchors_per_location = tf.shape(ws)[0]

        return tf.transpose(tf.stack([tf.zeros([num_anchors_per_location,]),
                                      tf.zeros([num_anchors_per_location,]),
                                      ws,hs]))

def make_anchors(base_anchor_size,anchor_scales,anchor_ratios,featuremap_height,featuremap_weight,stride,name = 'make anchors'):
    """

    :param base_anchor_size: base anchor size in different scales
    :param anchor_scales: anchor scales
    :param anchor_ratios: anchor ratios
    :param featuremap_height: height of featuremap
    :param featuremap_weight: width of featuremap
    :param stride:
    :param name:
    :return: anchors of shape [w*h*len(anchor_scales)*len(anchor_ratios),4]
    """
    base_anchor = tf.constant([0,0,base_anchor_size,base_anchor_size],dtype = tf.float32)
    base_anchors = enum_ratios(enum_scales(base_anchor,anchor_scales),anchor_ratios)

    _,_,ws,hs = tf.unstack(base_anchors,axis=1)

    x_center = tf.range(tf.cast(featuremap_weight,tf.float32),dtype=tf.float32) * stride
    y_center = tf.range(tf.cast(featuremap_height,tf.float32),dtype=tf.float32) * stride

    x_center,y_center = tf.meshgrid(x_center,y_center)
    ws,x_center = tf.meshgrid(ws,x_center)
    hs,y_center = tf.meshgrid(hs,y_center)

    box_centers = tf.stack([y_center,x_center],axis=2)
    box_centers = tf.reshape(box_centers,[-1,2])

    box_size = tf.stack([hs,ws],axis=2)
    box_size = tf.reshape(box_size,[-1,2])

    final_anchor = tf.concat([box_centers - 0.5+box_size,box_centers + 0.5 * box_size],axis=1)
    return final_anchor



if __name__ == '__main__':
    base_anchor = tf.constant([256],dtype = tf.float32)
    anchor_scales = tf.constant([1.0],dtype = tf.float32)
    anchor_ratios = tf.constant([0.5,1.0,2.0],dtype = tf.float32)

    sess = tf.Session()
    anchors = make_anchors(256,anchor_scales,anchor_ratios,38,50,16)
    _anchor = sess.run(anchors)
    print(_anchor.shape)










