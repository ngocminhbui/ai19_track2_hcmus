import tensorflow as tf
import numpy as np
import loss
def tf_argsort_des(inp_tensor):
    return tf.contrib.framework.argsort(inp_tensor,direction='DESCENDING')

def tf_count_view_freq(input_views, view_id_max):
    tf_cal_row_freq = lambda x : tf.histogram_fixed_width(x, (0,view_id_max),view_id_max + 1)
    input_views_freq = tf.map_fn(tf_cal_row_freq,input_views)
    input_views_freq_argsorted = tf.map_fn(tf_argsort_des,input_views_freq)
    return input_views_freq_argsorted

def tf_get_top1_view(batch_distances, gallery_views):
    tensor_gal_view = tf.convert_to_tensor(gallery_views, dtype=tf.int32)
    gal_view_id_max = np.max(gallery_views)

    dis_agr_sorted = tf.contrib.framework.argsort(batch_distances)
    dis_agr_sorted_view = tf.gather(tensor_gal_view,dis_agr_sorted)[:,:10]
    return tf_count_view_freq(dis_agr_sorted_view, gal_view_id_max)[:,0]

def tf_query_imgs(ins, gallery_embs, metric):    
    ins_distances = loss.cdist(ins, gallery_embs, metric = metric)
    return tf.reduce_mean(ins_distances, axis = 0)

def tf_query_img_in_same_view(view_mask, tensor_gal_embs, gallery_embs, metric):
    view_mask = tf.transpose(view_mask)
    view_mask = tf.cast(view_mask,tf.bool)
    ins = tf.boolean_mask(tensor_gal_embs,view_mask)
    return tf.cast(tf_query_imgs(ins, gallery_embs, metric),tf.float32)

def re_ranking_v2(top_1_view, predict_score, gallery_views, conf = 1.0): 
    #Get the top 1 view_id
    imp_view = [top_1_view]
    #Set images in important views:
    for j in range(len(imp_view)):
        imp_view_imgs = np.argwhere(gallery_views == imp_view[j])
        predict_score[imp_view_imgs] = conf
    return predict_score

def tf_query_extention(top1_view, gallery_views, gallery_embs, metric = 'euclidean'):
    tensor_gal_view = tf.convert_to_tensor(gallery_views, dtype=tf.int32)
    tensor_gal_embs = tf.convert_to_tensor(gallery_embs)
    #Create ones tensor with shape (batch_size x gallery_size) 
    tmp = tf.ones((tf.size(top1_view),tensor_gal_view.shape[0]), dtype=tf.int32)
    #Create tensor with value on each row equal the top 1 view
    tmp = tf.multiply(tmp, top1_view[:,None])
    tmp = tf.equal(tmp, tensor_gal_view)
    tmp = tf.cast(tmp,tf.float32)
    lmb_que_ext = lambda x: tf_query_img_in_same_view(x,tensor_gal_embs, gallery_embs, metric)
    return tf.map_fn(lmb_que_ext,tmp)