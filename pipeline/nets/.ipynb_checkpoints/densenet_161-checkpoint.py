import tensorflow as tf
import ipdb
from nets.dense_net import densenet161, densenet_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]

def endpoints(image, is_training, n_pids = None):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, 3]')

    image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

    with tf.contrib.slim.arg_scope(densenet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = densenet161(image, num_classes=n_pids, is_training=is_training)
    
    endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(endpoints['global_avg_pool'], [1, 2], name='pool5')
   
    return endpoints, 'densenet161'
