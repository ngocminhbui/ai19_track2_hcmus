import tensorflow as tf
from tensorflow.contrib import slim
import ipdb
def head(endpoints, embedding_dim, is_training, npids = 0):
    endpoints['head_output'] = slim.fully_connected(
        endpoints['model_output'], 1024, normalizer_fn=slim.batch_norm,
        normalizer_params={
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        })
    
    endpoints['emb_raw'] = slim.fully_connected(
        endpoints['head_output'], embedding_dim, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='emb')
    
    endpoints['emb'] = tf.nn.l2_normalize(endpoints['emb_raw'], -1, name="out_emb")
    
    endpoints['logits'] = slim.fully_connected(
        endpoints['head_output'], npids, activation_fn=None,
        weights_initializer=tf.orthogonal_initializer(), scope='logitis')
    
    return endpoints
