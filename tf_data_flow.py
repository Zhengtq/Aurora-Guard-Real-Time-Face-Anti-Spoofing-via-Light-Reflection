from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from glob import glob
import random
import numpy as np
import math
import cv2

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from PIL import Image, ImageEnhance
import PIL
from ztq_pylib.ztf_git.tf_img_process import *




#find_out_mask_module = tf.load_op_library('/home/ztq/user_ops/find_out_mask.so')
slim = tf.contrib.slim
DEPTH_IMG_SIZE = 32











def _load_batch_filename(input_queue, original_img_shape, channels, batch_size, all_image_num, to_height=224, to_width=224,img_norm=True, resize_type = 0):


        preprocess_threads = 4
        images_and_labels_list = []
	for _ in range(preprocess_threads):
		filenames, label, auxlabel= input_queue.dequeue()
		images = []
                image_depths = []
		for filename, single_label, single_auxlabel in zip(tf.unstack(filenames), tf.unstack(label), tf.unstack(auxlabel)):
		    file_contents = tf.read_file(filename)
    		    image = tf.image.decode_bmp(file_contents, channels=3)
    		    image = tf.reshape(image, original_img_shape)
                    image_depth = load_and_generate_depth_img(filename, original_img_shape, single_label)

    		    image, image_depth_p = _resize_crop_img(image, to_height, to_width, img_norm = img_norm, resize_type = resize_type, label = single_label, image_d = image_depth)
                    image = image[:, :, ::-1]

		    images.append(image)
                    image_depths.append(image_depth_p)

		images_and_labels_list.append([images, image_depths, label, auxlabel])




	img_batch, img_depth_batch,label_batch, auxlabel_batch = tf.train.batch_join(
		images_and_labels_list, batch_size=batch_size,
		shapes=[[to_height, to_width, channels], [DEPTH_IMG_SIZE,DEPTH_IMG_SIZE,1], (), ()], enqueue_many=True,
		capacity=4 * preprocess_threads * 100,
		allow_smaller_final_batch=True)

        img_batch = tf.reshape(img_batch, (batch_size, to_height, to_width, channels))
        img_depth_batch = tf.reshape(img_depth_batch, (batch_size, DEPTH_IMG_SIZE, DEPTH_IMG_SIZE, 1))
        label_batch = tf.reshape(label_batch, (batch_size, ))
        auxlabel_batch = tf.reshape(auxlabel_batch, (batch_size, ))



	img_batch = tf.identity(img_batch, 'image_batch')
	img_batch = tf.identity(img_batch, 'input')
	img_depth_batch = tf.identity(img_depth_batch, 'img_depth')
	label_batch = tf.identity(label_batch, 'label_batch')
	auxlabel_batch = tf.identity(auxlabel_batch, 'label_batch')

    	return img_batch, img_depth_batch, label_batch, auxlabel_batch





def _resize_crop_img(image, ran_crop_height, ran_crop_width,  img_norm = True , process_type = 'train', resize_type = 0, fast_mode = False, label = 1, image_d = None):

    

        image = tf.cast(image, tf.float32, name=None)
        if process_type == 'train':
            image_d = tf.image.resize_images(image_d, [DEPTH_IMG_SIZE, DEPTH_IMG_SIZE])
            pass

	return image, image_d



