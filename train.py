from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')
sys.path.append('./unet/')

import tf_data_flow as td
import tensorflow as tf
import numpy as np
import random
import unet as face_train
import math
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import slim
import cv2
from glob import glob
import os
import datetime
import operator
np.set_printoptions(precision=2, suppress=True)


RESIZE_TYPE = 10
IMG_NORM = False
RESTORE = True
USE_CENTER_LOSS = False
ALL_VAL_NUM = 170
split_time = 10
SPLIT_NUM = ALL_VAL_NUM // split_time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
GPU_NUM_ID = [0,1,2,3]
#GPU_NUM_ID = [0,1,2,3]
VAL_GPU_ID = 3
########## web_face setting ############320#
EPOC_NUM = 50
BATCH_SIZE = 32
BATCH_SIZE_SEP = BATCH_SIZE//len(GPU_NUM_ID)
TEST_SIZE = 128
RAN_CROP_SIZE = 128
SAMPLE_NUM = 25948
ORIGINAL_SIZE = [128, 128, 3]
CHANNELS = 3
LABEL_NUM = 1
validate_txt_root = './xxx.txt'
log_dir = '/xxx/'


LOSS_FUN = COM_LOSS()
ACC_FUN = COM_ACC()



def produce_lr(now_step, epoc_step):

    base_lr = 0.1
    if now_step < epoc_step:
        return 0.01
    else:
        now_lr = base_lr * math.pow(0.5, int((now_step - epoc_step) / (epoc_step * 3)))
        return now_lr




def train_cls():
 #   with tf.Graph().as_default():

            tf.logging.set_verbosity(tf.logging.INFO)

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)

            lr_in_use = tf.placeholder(tf.float32, shape=[])
            lr_in_use = tf.maximum(lr_in_use, 0.0001)
            optimizer = tf.train.MomentumOptimizer(lr_in_use, 0.9)

            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(len(GPU_NUM_ID)):

                    with tf.device('/gpu:%d' % GPU_NUM_ID[i]):
                        with tf.name_scope('%s_%d' % ('cnn_mg', i)) as scope:

                            images, images_depth, labels =  td._load_batch_filename(input_queue, ORIGINAL_SIZE, CHANNELS, BATCH_SIZE_SEP,SAMPLE_NUM,
                                RAN_CROP_SIZE, RAN_CROP_SIZE, img_norm = IMG_NORM, resize_type = RESIZE_TYPE)

                            origin_map, softmax_map, c_logits = face_train.inference(images,  num_classes=LABEL_NUM)
                            
                           
                            images_depth_show = images_depth
                            images_depth = tf.cast(images_depth, tf.int32)
                            images_depth = tf.squeeze(images_depth)
                            images_depth_one_hot = slim.one_hot_encoding(images_depth, 256)
                            map_loss = face_train.map_cross_entropy(images_depth_one_hot, softmax_map)

                            loss_softmax_sep_1 = LOSS_FUN._focal_loss_3(labels, c_logits)
                            loss_depth_sep = map_loss * 100
                            loss_regulation = LOSS_FUN._get_regulation_loss() * 0.01
                            accuracy_sep, _ = ACC_FUN._get_train_acc(c_logits, labels, label_num = 1)
                            loss_total_sep = loss_softmax_sep_1 + loss_depth_sep + loss_regulation 

                            tf.get_variable_scope().reuse_variables()
                            grads = optimizer.compute_gradients(loss_total_sep)
                            tower_grads.append(grads)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads = average_gradients(tower_grads)
                train_op = optimizer.apply_gradients(grads, global_step=global_step)


            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement=True
            with tf.Session(config = config) as sess:
                    make_rm_log_dir()
                    writer = tf.summary.FileWriter(log_dir,sess.graph)
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())


                    start_time = datetime.datetime.now()
                    epoc_step = int(SAMPLE_NUM/(BATCH_SIZE))
                    for loop in range(10000000):

  
                        _, g_step= sess.run([train_op, global_step], feed_dict={lr_in_use: now_lr})



if __name__=='__main__':
    train_cls()

