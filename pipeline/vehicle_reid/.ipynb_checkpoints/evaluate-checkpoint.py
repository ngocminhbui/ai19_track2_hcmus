import common
import loss
from importlib import import_module
from itertools import count
import os
import h5py
import json
import numpy as np
from sklearn.metrics import average_precision_score
from shutil import rmtree
import tensorflow as tf
from shutil import rmtree
import common
import loss

from vehicle_reid.query_extention import * 
from vehicle_reid.common import *

def calculate_ap(fid, pid_match, score):
    val_top = np.argsort(score)[-100:][::-1]
    ap = average_precision_score(pid_match[val_top], np.arange(100)) #core[val_top])
    try:
        k = np.where(pid_match[val_top])[0][0]
    except:
        print("Wrong!")
        k = 100
        ap = 0.0
    if np.isnan(ap):
        print()
        print("WARNING: encountered an AP of NaN!")
        print("This usually means a person only appears once.")
        print("In this case, it's because of {}.".format(fid))
        print("I'm excluding this person from eval and carrying on.")
        print()
    return ap, k

def save_test_img_index(result_folder,ques,aps, query_root):
    with open(os.path.join(result_folder,"index.csv"), "w") as fo:
        for i in range(len(aps)):
            query_img_path = os.path.join(query_root.split("/")[-2],ques[i])
            fo.write("{},{:.5f}\n".format(query_img_path,aps[i]))
    return

def save_predict_results(val_top, result_folder,score, pid, fid, pid_match, gallery_pids, gallery_fids, gallery_views, gal_root):
    #Missing images out of top 100:
    all_imgs = np.argwhere(gallery_pids == pid)
    found_imgs = val_top
    missing_mask = np.isin(all_imgs,found_imgs, invert=True)
    missing_imgs = all_imgs[missing_mask[:,0]][:,0]
    with open(os.path.join(result_folder, fid.replace('.jpg','.txt')), "w") as fo:
        for x in found_imgs:
            fo.write("{:s},{:5f},{},{}\n".format(os.path.join(gal_root,gallery_fids[x]),
                                                 score[x],
                                                 pid_match[x],
                                                 gallery_views[x]))
        for x in missing_imgs:
            fo.write("{:s},{:5f},{},{}\n".format(os.path.join(gal_root,gallery_fids[x]),
                                                 score[x],
                                                 pid_match[x],
                                                 gallery_views[x]))
        fo.close()

def save_submission(file, top100, gallery_fids):
    sub = [os.path.basename(x).split('.')[0] for x in gallery_fids[top100]]
    sub = [-1 if not x.isdigit() else x for x in sub]
    
    for img_id in sub[:-1]:
        file.write("{} ".format(int(img_id)))
    file.write('{}\n'.format(int(sub[-1])))
def get_value_from_h5_file(h5file, dataset):
    hf = h5py.File(h5file, 'r')
    return hf.get(dataset)

class Evaluator:
    def __init__(self, exp_root, gpu_id, batch_size = 128):
        self.exp_root = exp_root
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        
    def do_evaluate_with_config(self,config, query_extention = True):
        self.query_dataset = config.QUE_FILE
        self.gallery_dataset = config.GAL_FILE
        print("QUE: ",config.QUE_FILE)
        print("GAL: ",config.GAL_FILE)
        self.query_embeddings=os.path.join(
            self.exp_root,config.QUE_EMB_FILE)
        self.gallery_embeddings=os.path.join(
            self.exp_root,config.GAL_EMB_FILE)
        
        self.query_root = config.QUE_IMG_ROOT
        self.gal_root   = config.GAL_IMG_ROOT
        self.result_folder = config.RESULTS_ROOT
        
        self.gal_view_point = os.path.join(self.exp_root, config.GAL_VIEW_POINT)
        self.que_view_point = os.path.join(self.exp_root, config.QUE_VIEW_POINT)
        
        #Remove result folder:
        if (os.path.exists(self.result_folder)):
            rmtree(self.result_folder)
        os.mkdir(self.result_folder)
        
        self.load_embed_files()
        self.exe_query(query_extention)
    def calculate_distances(self, Q, G):
        metric = 'euclidean'
        batch_embs = tf.data.Dataset.from_tensor_slices(
        (Q)).batch(self.batch_size).make_one_shot_iterator().get_next()
        batch_distances = loss.cdist(batch_embs, G , metric=metric)
        distances = np.zeros((len(Q), len(G)), np.float32)
        with tf.Session() as sess:
            for start_idx in count(step=self.batch_size):
                try:
                    dist = sess.run(batch_distances)
                    distances[start_idx:start_idx + len(dist)] = dist
                except tf.errors.OutOfRangeError:
                    print()  # Done!
                    break
        print(distances.shape)
        return distances

    def load_embed_files(self):
        print("Load: ", self.query_dataset)
        self.query_pids, self.query_fids, self.query_views = common.load_dataset(self.query_dataset, None)
        
        print("Load: ", self.gallery_dataset)
        self.gallery_pids, self.gallery_fids, self.gallery_views = common.load_dataset(self.gallery_dataset, None)
        
        self.gallery_views = self.gallery_views.astype(int)
        self.query_views = self.query_views.astype(int)
        print("Load: ", self.query_embeddings)
        with h5py.File(self.query_embeddings, 'r') as f_query:
            self.query_embs = np.array(f_query['emb'])
        print("Load: ", self.gallery_embeddings)
        with h5py.File(self.gallery_embeddings, 'r') as f_gallery:
            self.gallery_embs = np.array(f_gallery['emb'])

        query_dim = self.query_embs.shape[1]
        gallery_dim = self.gallery_embs.shape[1]
        if query_dim != gallery_dim:
            raise ValueError('Shape mismatch between query ({}) and gallery ({}) '
                             'dimension'.format(query_dim, gallery_dim))
        print("==========================")
    
    def select_top(self, inp_arr, selected):
        for i in np.argsort(inp_arr):
            if (i not in selected):
                selected.append(i)
                return selected, i
    def select_NN(self, top1_view, tracklet_dists, k = 4):
        selected = [top1_view]
        selected, top2_view = self.select_top(tracklet_dists[top1_view], selected)
        selected, top3_view = self.select_top(tracklet_dists[top2_view], selected)
        selected, top4_view = self.select_top(tracklet_dists[top3_view], selected)
        return selected 
    
    def track_re_ranking(self, top_list, tracklet_mapper, score):
        tmp = score
        for i, top_track in enumerate(top_list):
            tmp = re_ranking_v2(tracklet_mapper[top_track], tmp, self.gallery_views, 1.0 - 0.05 * i)
        return tmp
        
    def exe_query(self, query_extention = True):
        aps = []
        ques = []
        cmc = np.zeros(len(self.gallery_pids), dtype=np.int32)
        gallery_views_id, gallery_views_count = np.unique(self.gallery_views, return_counts=True)
        
        metric = 'euclidean'
        print(self.gallery_embs.shape)
        print(self.query_embs.shape)
        batch_pids, batch_fids, batch_embs = tf.data.Dataset.from_tensor_slices(
        (self.query_pids, self.query_fids, self.query_embs)).batch(self.batch_size).make_one_shot_iterator().get_next()
        batch_distances = loss.cdist(batch_embs, self.gallery_embs, metric=metric)
        
        self.submission_file = "track2.txt"
        print("Total queries: ", len(self.query_fids))
        print("Results folder: ", self.result_folder)
        print("Submission file: ", self.submission_file)
        
        
        dist_h5_file = "results_dists/test798x798.h5"
        tracklet_dists = load_h5(dist_h5_file)
        tracklet_mapper = load_h5(dist_h5_file, "mapper")
        trklet_dict = {}
        
        for i, trid in enumerate(tracklet_mapper):
            trklet_dict[i] = i
        
        with tf.Session() as sess, open(self.submission_file, "w") as f_sub:
            for start_idx in count(step=self.batch_size):
                try:
                    if (query_extention):
                        top1_view = tf_get_top1_view(batch_distances, self.gallery_views)
                        que_ext_re_ranking = tf_query_extention(top1_view, self.gallery_views, self.gallery_embs)
                        top1_views, distances, pids, fids = sess.run([top1_view, que_ext_re_ranking, batch_pids, batch_fids])
                    else:
                        distances, pids, fids = sess.run([batch_distances, batch_pids, batch_fids])
                        top1_view = np.zeros(fids.shape, dtype=int)

                    print('\rCalculating batch {}-{}/{}'.format( start_idx, start_idx + len(fids), len(self.query_fids)), flush=True, end='')

                except tf.errors.OutOfRangeError:
                    print()  # Done!
                    break

                pids, fids = np.array(pids, '|U'), np.array(fids, '|U')
                pid_matches = self.gallery_pids[None] == pids[:,None]
                scores = 1 / (1 + distances)
                
                for i in range(len(distances)):
                    fid = fids[i]
                    pid = pids[i]
                    pid_match = pid_matches[i,:]
                    score = scores[i]
                    top1_view = top1_views[i]
                    top1_view = trklet_dict[top1_view]
                    if(query_extention):
                        selected = self.select_NN(top1_view, tracklet_dists)
                        score = self.track_re_ranking(selected, tracklet_mapper, score)
                    
                    top100 = np.argsort(score)[-100:][::-1]
                    #Save submission file
                    save_submission(f_sub, top100, self.gallery_fids)
                    #Save predict results:
                    save_predict_results(top100, self.result_folder,score, pid, fid, pid_match, self.gallery_pids, self.gallery_fids, self.gallery_views, self.gal_root.split("/")[-2])
                    
                    #Calculate AP:
                    ap, k = calculate_ap(fid, pid_match, score)
                    cmc[k:] += 1
                    aps.append(ap)
                    ques.append(fid)

            # Save index.csv
            save_test_img_index(self.result_folder,ques,aps, self.query_root)  

            # Compute the actual cmc and mAP values
            cmc = cmc / len(self.query_pids)
            mean_ap = np.mean(aps)
            print('mAP: {:.2%} | top-1: {:.2%} top-2: {:.2%} | top-5: {:.2%} | top-10: {:.2%}'.format(
                mean_ap, cmc[0], cmc[1], cmc[4], cmc[9]))
            
            
            