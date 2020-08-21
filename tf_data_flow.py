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





def _load_batch_t(file_root, original_img_shape, channels, batch_size, all_image_num, to_height=224, to_width=224,img_norm=True, resize_type = 0):


    filename_queue = tf.train.string_input_producer([file_root])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    single_example = tf.parse_single_example(serialized_example,
            features = {
                'image/encoded':tf.FixedLenFeature([], tf.string),
                'image/class/label':tf.FixedLenFeature([], tf.int64)
          #      'image/m_label':tf.FixedLenFeature([], tf.int64)
                })


    label = single_example['image/class/label']
    img = tf.decode_raw(single_example['image/encoded'], tf.uint8)


    img = tf.reshape(img, original_img_shape)
    img = img[:, :, ::-1]
    img = _resize_crop_img(img, to_height, to_width, img_norm = img_norm, resize_type = resize_type, label = label)
    img = img[:, :, ::-1]
#    label = single_example['image/m_label']



#    min_after_dequeue = int(all_image_num * 0.4)
    num_threads =1
    min_after_dequeue = 1500
    capacity = min_after_dequeue + (num_threads+1) * batch_size

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = batch_size,
                                                    capacity = capacity,
                                                    num_threads =num_threads,
                                                    min_after_dequeue = min_after_dequeue)


 #     img_batch, label_batch = tf.train.batch([img, label],
                                                    #  num_threads=num_threads,
                                                    #  batch_size = batch_size,
                                                    #  capacity = min_after_dequeue  + num_threads * batch_size)


    return img_batch, label_batch


def load_and_generate_depth_img(origin_img_root, original_img_shape, label):



    def file_exist_fun(filename):
        return os.path.exists(filename)


    def load_depth_img(origin_img_root, original_img_shape):
        filename_1 = tf.regex_replace(origin_img_root, ".bmp", "_prn_depth.jpg")
        filename_2 = tf.constant('/home/ztq/project/antispoof/dl_project/others/avg_depth.jpg', tf.string)


        result = tf.py_func(file_exist_fun, [filename_1], tf.bool)

        filename = tf.cond(result, lambda: filename_1, lambda: filename_2)



        file_contents = tf.read_file(filename)

        image =tf.image.decode_jpeg(file_contents, channels=1)
        image = tf.reshape(image, [original_img_shape[0], original_img_shape[1], 1])
        image = tf.cast(image, tf.float32)
        return image

    def generate_zero_img(original_img_shape):
        no_depth_img = tf.zeros([original_img_shape[0], original_img_shape[1], 1])
        return no_depth_img

    image = tf.cond(label > 0, lambda: generate_zero_img(original_img_shape), lambda: load_depth_img(origin_img_root, original_img_shape))
 #   image = load_depth_img(origin_img_root, original_img_shape)
 #   image = tf.image.central_crop(image,0.5)
 #   image = tf.image.resize_images(image, [32, 32])

    return image






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

def _generate_list(img_roots_txt, batch_size, epoc_size):



	image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
	labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
	auxlabel_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')

	input_queue = data_flow_ops.FIFOQueue(capacity=3900000,
		                    dtypes=[tf.string, tf.int64, tf.int64],
		                    shapes=[(1,), (1,),(1,)],
		                    shared_name=None, name=None)
	enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, auxlabel_placeholder], name='enqueue_op')

	input_file = open(img_roots_txt, 'r')
	img_root_list = []
	label_list = []
        auxlabel_list = []
	for item in input_file:
		item = item.strip().split('####')
		img_root_list.append(item[0])
		label_list.append(int(item[1]))
                auxlabel_list.append(int(item[2]))
	all_num = len(label_list)


        index_queue = tf.train.range_input_producer(all_num, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        index_dequeue_op = index_queue.dequeue_many(all_num - 10, 'index_dequeue')

	return input_queue, enqueue_op, img_root_list, label_list, auxlabel_list, index_dequeue_op, image_paths_placeholder, labels_placeholder, auxlabel_placeholder



def _resize_crop_img(image, ran_crop_height, ran_crop_width,  img_norm = True , process_type = 'train', resize_type = 0, fast_mode = False, label = 1, image_d = None):

        def tf_resize_and_compress(image, the_range):

            def inner_fun_rc(image, the_range):
            #    image = tf_ran_resize_tur123(image, resize_ratio = 0.5, ran_ratio = 0.5)
                image = tf_ran_jpeg_compress(image, 100, the_range = the_range) 
                return image

            ran = tf.random_uniform([])
            image = tf.cond(ran<0.02, lambda:inner_fun_rc(image, the_range), lambda:image)
            return image


        def tf_up_contrast_sa(image, ran_ratio = 0.05):
            
            def inner_fun_ca(image):
                image = tf.image.random_saturation(image, lower=1.0, upper=1.5)
                image = tf.image.random_contrast(image, lower=1.5, upper=2.0)
                return image
            
            ran = tf.random_uniform([])
            image = tf.cond(ran<ran_ratio, lambda:inner_fun_ca(image), lambda:image)

            return image




        def tur_ratio_pos(image, ratio=1.0):


            def pos_norm_tur(image):
                image = ran_img_color_tur(image, ran_ratio = 0.1, ratio = 0.2)
                image = random_color_hue(image,ran_ratio = 0.1, ratio=0.2)
                image = g_noise_tur(image, ran_ratio = 0.1, ratio = 0.5)
                image = gray_color_tur(image, ran_ratio = 0.04)
                image = all_yuv_tur(image, ran_ratio = 0.05, ratio = 0.4)
                image = tf_ran_resize_tur123(image, ran_ratio = 0.1)
                image = tf_ran_avgfilter(image, ran_ratio = 0.03, class_num = 4)
                image = ran_img_motion_blur(image, 0.03, 3, 25)
                image = generate_color_dis_pos(image,ran_ratio = 0.03)
                image = tf_up_contrast_sa(image)
                return image


            def pos_abnorm_tur(image):
                image = ran_img_color_tur(image, ran_ratio = 0.1, ratio = 0.2 * 5)
                image = random_color_hue(image,ran_ratio = 0.1, ratio=0.2)
                image = g_noise_tur(image, ran_ratio = 0.1, ratio = 0.5)
                image = gray_color_tur(image, ran_ratio = 0.04)
                image = all_yuv_tur(image, ran_ratio = 0.05, ratio = 0.4 * 5)
                image = tf_ran_resize_tur123(image, ran_ratio = 0.1)
                image = tf_ran_avgfilter(image, ran_ratio = 0.03, class_num = 4)
                image = ran_img_motion_blur(image, 0.03, 3, 25)
                image = generate_color_dis_pos(image,ran_ratio = 0.03)
                image = tf_up_contrast_sa(image)
                return image

            ran = tf.random_uniform([])
            image = tf.cond(ran<0.95, lambda:pos_norm_tur(image), lambda:pos_abnorm_tur(image))

            return image

        def tur_ratio_neg(image, ratio=1.0):
             image = ran_img_color_tur(image, ran_ratio = 0.1, ratio = 0.2)
             image = random_color_hue(image,ran_ratio = 0.1, ratio=0.2)
         #    image = g_noise_tur(image, ran_ratio = 0.1, ratio = 0.5)
         #    image = all_yuv_tur(image, ran_ratio = 0.05, ratio = 0.4)
         #    image = tf_ran_resize_tur123(image, ran_ratio = 0.1)
         #    image = tf_ran_avgfilter(image, ran_ratio = 0.03, class_num = 1)
         #    image = ran_img_motion_blur(image, 0.03, 3, 10)
        #     image = tf_lower_img_bright(image, ran_ratio = 0.1, bound = [-50, 0])
             return image



        image = tf.cast(image, tf.float32, name=None)
        if process_type == 'train':
            image_d = tf.image.resize_images(image_d, [DEPTH_IMG_SIZE, DEPTH_IMG_SIZE])
            pass
#              image = tf.cond(label > 0, lambda: ran_face_size_tur_3(image, ran_ratio=0.1), lambda: ran_face_size_tur_3(image, ran_ratio=0.02))
            #  image = tf.concat([image, image_d], axis=2)
    #  #        ran = tf.random_uniform([])
    #  #        image = tf.cond(ran<0.8, lambda: ran_face_size_tur_2(image) , lambda:ran_face_size_tur_3(image))

            #  image = ran_face_size_tur_3(image)
            #  image = tf_ran_rot90(image)
            #  image = tf.image.random_flip_left_right(image)
            #  image = tf.cond(label > 0, lambda: tf_random_crop(image, 240, 320, 280, channel=4),
                    #  lambda: tf_random_crop(image, 240, 320, 280, channel=4))

            #  image = tf.cond(label > 0, lambda: transform_perspective(image), lambda: transform_perspective(image, ran_ratio=0.01))
            #  image = tf_rotate_img(image, ran_ratio = 0.2)

            #  image_split = tf.split(image,4, axis=2)
            #  image = tf.concat(image_split[:3], axis =2)
            #  image_d = image_split[-1]
            #  image_d = tf.image.resize_images(image_d, [DEPTH_IMG_SIZE, DEPTH_IMG_SIZE])

            #  black_mask = tf.not_equal(image, 0)
            #  black_mask = tf.cast(black_mask, tf.float32)

            #  image = tf.cond(label > 0, lambda: tur_ratio_pos(image, 0.8), lambda: tur_ratio_neg(image, 0.8))
            #  image = tf.reshape(image, (280, 280, 3))
#  #              image = tf.cond(label > 0, lambda: tf_resize_and_compress(image, [1, 100]), 
                    #  #  lambda: tf_resize_and_compress(image, [50, 100]))
            #  image = tf.cond(label > 0, lambda: tf_resize_and_compress(image, [1, 100]), 
                    #  lambda: image)
            #  image = tf.multiply(image, black_mask)
            #  image = tf.clip_by_value(image, 0.0, 255.0)

	if img_norm:
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)


       #   if height and width:
		#  # Resize the image to the specified height and width.

                #  if process_type == 'train':
		    #  image = tf.expand_dims(image, 0)

                #  if resize_type == 0:
		    #  image = tf.image.resize_bilinear(image, [height, width],  align_corners=False)
                #  if resize_type == 1:
		    #  image = tf.image.resize_nearest_neighbor(image, [height, width],  align_corners=False)


                #  if process_type == 'train':
		    #  image = tf.squeeze(image, [0])

        if img_norm:
	    image = tf.subtract(image, 0.5)
	    image = tf.multiply(image, 2.0)

	return image, image_d



