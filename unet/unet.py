import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def head_stem(x):

    with tf.variable_scope('head_stem'):
        x = slim.conv2d(x, 32, (3, 3), stride=2, scope='Conv1')
        x = slim.conv2d(x, 64, (3, 3), stride=2, scope='Conv2')


    return x






def inverted_residual_block(x, block_num=0, exp_ratio = 3, head_channels_z = None):

    
    in_channels = x.shape[3].value
    if head_channels_z == None:
        head_channels = in_channels * 2
    else:
        head_channels = head_channels_z
   
    
    with tf.variable_scope('pre_' + str(block_num)):
        x = slim.conv2d(x, head_channels, (1, 1), stride=1, scope='conv1_pre')


    block_input = x
    with tf.variable_scope('inverted_rb_' + str(block_num)):
        exp_channels = exp_ratio * head_channels
        out_channels = head_channels
        x = slim.conv2d(x, exp_channels, (1, 1), stride=1, scope='conv1')
        x = depthwise_conv(x, kernel=3, stride=1 , scope='depthwise1')
        x = slim.conv2d(x, out_channels, (1, 1), stride=1, activation_fn=None, scope='conv2')
        x = block_input + x
    
    return x






def inference(images, is_training=True, num_classes=2, depth_multiplier='1.0', reuse=None):


    out_data = {}
    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = images

    all_collect_map = []
    with tf.variable_scope('Unet', reuse=tf.AUTO_REUSE):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        #wd=0.00018
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):
          with slim.arg_scope([slim.conv2d],
                  weights_regularizer=slim.l2_regularizer(0.000005)):

            x = head_stem(x)
            x = slim.conv2d(x, 128, (3, 3), stride=1, scope='Conv1')

            down_channel_collect = []
            for down_ind in range(5):
                if down_ind == 0:
                    x = inverted_residual_block(x, block_num = down_ind, head_channels_z = 64)
                else:
                    x = inverted_residual_block(x, block_num = down_ind)

                if down_ind != 4:
                    down_channel_collect.append(x)
                if down_ind != 4:
                    x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')


            x = tf.image.resize_images(x, [x.shape[1].value * 2, x.shape[2].value * 2])
            x = slim.conv2d(x, 512, (1, 1), stride=1, scope='conv2_middle')


            x = tf.concat([x,down_channel_collect[-1]], axis=-1)
            x = inverted_residual_block(x, block_num = 5, head_channels_z = 256)
            x = tf.image.resize_images(x, [x.shape[1].value * 2, x.shape[2].value * 2])

            x = tf.concat([x,down_channel_collect[-2]], axis=-1)
            x = inverted_residual_block(x, block_num = 6, head_channels_z = 128)
            x = tf.image.resize_images(x, [x.shape[1].value * 2, x.shape[2].value * 2])

            x = tf.concat([x,down_channel_collect[-3]], axis=-1)
            x = inverted_residual_block(x, block_num = 7, head_channels_z = 64)
            x = tf.image.resize_images(x, [x.shape[1].value * 2, x.shape[2].value * 2])

            x = tf.concat([x,down_channel_collect[-4]], axis=-1)
            x = inverted_residual_block(x, block_num = 8, head_channels_z = 64)

            x = slim.conv2d(x, 256, (3, 3), stride=1, scope='out', normalizer_fn=None, activation_fn=None)
            out_data['out_put'] = x
          

            softmax_map = pixel_wise_softmax(x)
            c_logits = C_net(x)

            x = tf.identity(x, 'Zoutput')
            

    return x, softmax_map, c_logits



def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def C_net(softmax_map):


    with tf.variable_scope('Cnet_'):
        softmax_map = tf.reduce_max(softmax_map, axis=3, keepdims=True)
        x = slim.conv2d(softmax_map, 16, (3, 3), stride=2, scope='Conv1')
        x = slim.conv2d(x, 8, (3, 3), stride=2, scope='Conv2')
        x = slim.conv2d(x, 4, (3, 3), stride=2, scope='Conv3')

        x = slim.flatten(x, scope='flatten')
        c_logits = slim.fully_connected(x, 1, normalizer_fn=None, activation_fn=None)


    return c_logits 


def map_cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")




@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x
