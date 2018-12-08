#coding:utf-8
from __future__ import absolute_import,division,print_function

import numpy as np
import cv2
from FPN_tensorflow.help_utils.help_utils import show_boxes_in_img

def make_anchors(base_anchor_size,anchor_scales,anchor_ratios,featuremaps_height,featuremap_weight,stride):
    """

    :param base_anchor_size:
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremaps_height:
    :param featuremap_weight:
    :param stride:
    :return: anchor of shape:[w*h*9,4]
    """
    base_anchor = [0,0,base_anchor_size,base_anchor_size] #[y_center,x_center,h,w]
    per_location_anchors = enum_ratios(enum_scales(base_anchor,anchor_scales),anchor_ratios)
    print(per_location_anchors.shape)
    ws,hs = per_location_anchors[:,2],per_location_anchors[:,3]
    x_centers = np.arange(featuremap_weight) * stride
    y_centers = np.arange(featuremaps_height) * stride
    x_centers,y_centers = np.meshgrid(x_centers,y_centers)

    ws,x_centers = np.meshgrid(ws,x_centers)
    hs,y_centers = np.meshgrid(hs,y_centers)
    box_centers = np.stack([y_centers,x_centers],axis=2)
    box_centers = np.reshape(box_centers,[-1,2])

    box_sizes = np.stack([hs,ws],axis=2)
    box_sizes = np.reshape(box_sizes,[-1,2])

    final_anchors = np.concatenate([box_centers-0.5*box_sizes,box_centers+0.5*box_sizes],axis=1)
    final_anchors = final_anchors.astype(dtype=np.float32)
    print("this is make anchors")
    return final_anchors







def enum_scales(base_anchor,anchor_scales):
    """

    for base anchor:center point is zero
    :param base_anchor: [y_center,x_center,h,w] --> may[0,0,256,256]
    :param anchor_scales: maybe [0,5,1.0,2.0]
    :return:
    """
    base_anchor = np.array(base_anchor)
    anchor_scales = np.array(anchor_scales).reshape(len(anchor_scales),1)

    return base_anchor * anchor_scales

def enum_ratios(anchors,anchor_ratios):
    """
    h/w = ratio
    :param anchors:
    :param anchor_ratios:
    :return:
    """
    ws = anchors[:,3] # for base anchor w == h
    hs = anchors[:,2]
    sqrt_ratios = np.sqrt(np.array(anchor_ratios))
    ws = np.reshape(ws / sqrt_ratios[:,np.newaxis],[-1])
    hs = np.reshape(hs * sqrt_ratios[:,np.newaxis],[-1])
    assert ws.shape == hs.shape ,'h shape is not equal w shape'

    num_anchors_per_location = ws.shape[0]
    print(num_anchors_per_location)
    return np.hstack([np.zeros((num_anchors_per_location,1)),
                     np.zeros((num_anchors_per_location,1)),
                     ws[:,np.newaxis],
                     hs[:,np.newaxis]])


def filter_outside_boxes(anchors,img_h,img_w):
    """

    :param anchors:[-1,4]...[ymin,xmin,ymax,xmax]
    :param img_h:
    :param img_w:
    :return:
    """

    index = (anchors[:,0] > 0) & (anchors[:,0] < img_h) & \
            (anchors[:, 1] > 0) & (anchors[:, 1] < img_w) & \
            (anchors[:, 2] > 0) & (anchors[:, 2] < img_h) & \
            (anchors[:, 3] > 0) & (anchors[:, 3] < img_w)

    valid_indices = np.where(index == True)[0]

    return valid_indices
def show_anchors_in_img(anchors):
    img = cv2.imread("1.jpg")
    img = cv2.resize(img,(800,600),interpolation=cv2.INTER_AREA)
    img = show_boxes_in_img(img,anchors)
    cv2.imshow('resize_img',img)
    cv2.waitKey(0)


if __name__ == "__main__":
    print(enum_scales([0,0,256,256],[0.5,1.,2.0]))
    print("________")
    anchors = make_anchors(256,[1.0],[0.5,1.,2.],featuremaps_height=38,featuremap_weight=50,stride=16)
    indices = filter_outside_boxes(anchors,img_h=600,img_w=800)
    show_anchors_in_img(np.column_stack([anchors[indices],np.ones(shape=(anchors[indices].shape[0],1))]))



































































































