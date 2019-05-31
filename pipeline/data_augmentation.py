import tensorflow as tf

def gramma_augmentation(tf_img):
    rnd_gramma = 0.4 + tf.random.uniform([1],-0.1,0.1)
    gramma_img = tf.image.adjust_gamma(tf.cast(tf_img, tf.float32), gamma=rnd_gramma[0], gain=0.1)
    gramma_img = tf.clip_by_value(gramma_img,0.0, 1.0) * 255.0
    return gramma_img