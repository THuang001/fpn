#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

############################
#        dataset
############################
tf.app.flags.DEFINE_string('dataset_tfrecord','../data/tfrecords','tfrecord of fruits dataset')

tf.app.flags.DEFINE_integer('new_img_size',224,'the value of new height and new widtd,new_height = new_width')


############################
#        data batch
############################
tf.app.flags.DEFINE_integer('num_class',134,'num of classes')
tf.app.flags.DEFINE_integer('batch_size',32,'num of imgs in a batch')
tf.app.flags.DEFINE_integer('val_bathc_size',8,'val of test batch')

############################
#        learning rate
############################
tf.app.flags.DEFINE_float('lr_begin',0.001,'the value of learning rate start with')
tf.app.flags.DEFINE_integer('decay_steps',3000,"after 'decay steps' steps,learning rate begin decay")
tf.app.flags.DEFINE_float('decay_rate',0.1,'decay rate')

############################
#      optimizer-- MomentumOptimizer
############################
tf.app.flags.DEFINE_float('momentum',0.9,'accumulation =momentum * accumulation + gradient')

############################
#      train
############################
tf.app.flags.DEFINE_integer('max_steps',20010,'max iterate steps')
tf.app.flags.DEFINE_string('pretraine_model_path','../data/pretrained_weights/vgg_16.ckpt','the path of pretrained weights')
tf.app.flags.DEFINE_float('weight_decay',0.0005,'weight_decay in regulation')

############################
#  summary and save_weights_checkpoint
############################
tf.app.flags.DEFINE_string('summary_path','../output/vgg16_summary','the path of summary write to')
tf.app.flags.DEFINE_string('trained_checkpoint','../output/vgg16_trainedweights','the path to save trained_weights')

FLAGS = tf.app.flags.FLAGS
