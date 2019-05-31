from vehicle_reid import exp_config
import sys
sys.path.append("../")
from classifier import classifier_common as clscom
import numpy as np
import h5py
import csv
import os.path as osp
from matplotlib import pyplot as plt

TRAIN_ROOT = "/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
TEST_ROOT  = "/home/hthieu/AICityChallenge2019/data/Track2Data/image_test/"
QUERY_ROOT =  "/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
VEHITYPE = "/home/hthieu/AICityChallenge2019/triplet-reid/vehicle_type_prediction"

TEST_TRUCK = {
    'root': TEST_ROOT,
    'csv_in': "data/vehi_type_test/3.csv",
    "out_h5": "query_ext/test_bus.h5",
    "dist": "test_bus_dist",
    "bus2type": "query_ext/test_bus_3_type.h5"
}
TRAIN_TRUCK = {
    'root': TRAIN_ROOT,
    'csv_in': "data/vehi_type/3.csv",
    "out_h5": "query_ext/train_bus.h5",
    "dist": "train_bus_dist",
    "bus2type": "query_ext/train_bus_3_type.h5"
}
TRAIN_PICKUP = {
    'root': TRAIN_ROOT,
    'csv_in': "data/vehi_type/1.csv",
    "out_h5": "query_ext/train_pickup.h5",
    "dist": "train_pickup_dist"  
}
QUERY = {
    'root': QUERY_ROOT,
    'csv_in': "data/track2_query.csv",
    "out_h5": "query_ext/query_all.h5",
    "dist": "query_dist",
}
TESTFULL = {
    'root': TEST_ROOT,
    'csv_in': "data/track2_test_v3.csv",
    "out_h5": "query_ext/test_all.h5",
    "dist": "test_all_dist",
}

TRAINCROP = {
    'root': "/home/hthieu/AICityChallenge2019/data/Track2Data/image_train_crop/",
    'csv_in': "data/track2_train_v3.csv",
    "out_h5": "query_ext/train_all_crop.h5",
    "dist": "train_all_dist",
}

TESTCROP = {
    'root': "/home/hthieu/AICityChallenge2019/data/Track2Data/image_test_crop/",
    'csv_in': "data/track2_test_v3.csv",
    "out_h5": "query_ext/test_all_crop.h5",
    "dist": "test_all_dist",
}

QUERYCROP = {
    'root': "/home/hthieu/AICityChallenge2019/data/Track2Data/image_query_crop/",
    'csv_in': "data/track2_query.csv",
    "out_h5": "query_ext/query_all_crop.h5",
    "dist": "query_dist",
}

TRAINFULL = {
    'root': TRAIN_ROOT,
    'csv_in': "data/track2_train_v3.csv",
    "out_h5": "query_ext/train_all.h5",
    "dist": "train_all_dist",
}

TEST_BUS = {
    'root': TEST_ROOT,
    'csv_in': "data/vehi_type_test/2.csv",
    "out_h5": "query_ext/test_bus.h5",
    "dist": "test_bus_dist",
    "bus2type": "query_ext/test_bus_2_type.h5"
}
TRAIN_BUS = {
    'root': TRAIN_ROOT,
    'csv_in': "data/vehi_type/2.csv",
    "out_h5": "query_ext/train_bus.h5",
    "dist": "train_bus_dist",
    "bus2type": "query_ext/train_bus_2_type.h5"
}


TEST_WHEEL = {
    'root': "/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/part/wheel/test" ,
    'csv_in': "/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/part/csv_file/wheel_test.csv",
    "out_h5": "query_ext/wheel_test.h5",
    "dist": "test_wheel_dist"
}

QUERY_WHEEL = {
    'root': "/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/part/wheel/query",
    'csv_in': "/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/part/csv_file/wheel_query.csv",
    "out_h5": "query_ext/wheel_query.h5",
    "dist": "query_wheel_dist"
}


def EmbedTest(embedder):    
    config = exp_config.Test()
    embedder.do_experiment_with_config(config)
    
def EmbedValidate(EXP_DIR, embedder):
    config = exp_config.Validate()
    config.QUE_EMB_FILE = osp.join(EXP_DIR, config.QUE_EMB_FILE)
    config.GAL_EMB_FILE = osp.join(EXP_DIR, config.GAL_EMB_FILE)
    embedder.do_experiment_with_config(config)
    
def EmbedCustomEmbed(embedder):
    config = exp_config.CustomEmbed()
    embedder.do_experiment_with_config(config)
    
def EvaluateTest(evaluator):
    config = exp_config.Test()
    evaluator.do_evaluate_with_config(config)

def EvaluateValidate(evaluator):
    config = exp_config.Validate2()
    evaluator.do_evaluate_with_config(config)
    
def get_masks(ids):
    same_id = ids[:,None] == ids[None,:]
    positive_mask = np.logical_xor(same_id, np.eye(same_id.shape[0], dtype=np.bool))
    negative_mask = np.logical_not(same_id)
    tri = np.tril(np.ones(same_id.shape, dtype=np.bool))
    return np.logical_and(positive_mask, tri), np.logical_and(negative_mask, tri)

def save_dist(dist, out_file, mapper=None):
    with h5py.File("results_dists/{}.h5".format(out_file), "w") as f:
        f.create_dataset("distances", data=dist)
        if (mapper != None):
            f.create_dataset("mapper", data = mapper)
        f.close()
    
    with open("results_dists/{}.txt".format(out_file),"w") as fo:
        for x in dist:
            for y in x:
                fo.write("{:.5f} ".format(y))
            fo.write("\n")
        fo.close()

def save_txt(dist, out_file):
     with open(out_file,"w") as fo:
        for x in dist:
            for y in x:
                fo.write("{:.5f} ".format(y))
            fo.write("\n")
        fo.close()
        
        
def load_h5(inp_file, dataset = 'distances'):
    with h5py.File(inp_file, "r") as f:
        tmp = np.array(f[dataset])
        f.close()
    return tmp


def load_txt_float32(inp_file):
    return np.loadtxt(inp_file)

import cv2
def read_csv_dataset(inp_file):
    with open(inp_file) as fi:
        csv_reader = csv.reader(fi, delimiter = ',')
        all_ids = []
        all_imgs= []
        all_tlet = []
        for info in csv_reader:
            all_ids.append(int(info[0]))
            all_imgs.append(osp.basename(info[1]))
            if (len(info) == 3):
                all_tlet.append(info[2])
            else:
                all_tlet.append(0)
        return np.array(all_ids), np.array(all_imgs), np.array(all_tlet).astype(np.int32)
    
def vis_img_group(imgs_path, columns = 10, rows = 10):
    w=10
    h=10
    fig=plt.figure(figsize=(16, 16))
    fig.tight_layout()
    for i in range(1, min(columns*rows +1, len(imgs_path)+1)):
        img = plt.imread(imgs_path[i-1])
        img = cv2.resize(img,(255,255))
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
def vis_img(img_path):
    img = plt.imread(img_path)
    plt.imshow(img)
    
def load_emb_h5(file_path):
    f = h5py.File(file_path, 'r')
    _emb = np.array(f['emb'])
    _id  = np.array(f['id'])
    _fol = np.array(f['fol_id'])
    f.close()
    return _emb, _id, _fol

def get3avt(file):
    avt_imgs={}
    with open("../data/Track2Data/{}_track.txt".format(file)) as fi:
        for i, info in enumerate(fi):
            info = info.strip().split(' ')
            avts = [info[0]]
            if (len(info)<2):
                avts.append(info[0])
            else:
                avts.append(info[len(info)//2])
            avts.append(info[-1])
            avt_imgs[i] = avts
        fi.close()
    return avt_imgs
    
def read_bus_query_test():
    _, query_id, query_score = clscom.read_csv_file("vehicle_type_prediction/030519_query_type_0.92.csv")
    _, test_id, test_score = clscom.read_csv_file("vehicle_type_prediction/030519_test_type_0.92.csv")
    query_bus = clscom.apply_threshold(query_score, 2, 0.99)
    
    test_full_emb, test_full_id, test_full_track = load_emb_h5(TESTFULL['out_h5'])
    test_full_uni, tefuid = np.unique(test_full_track,  return_index=True)
    test_bus = np.array([clscom.get_majority(test_full_track,x,test_score) for x in test_full_uni])
    test_bus  = clscom.apply_threshold(test_bus, 2, 0.99)
    return query_bus, test_bus 

def read_truck_query_test():
    _, query_id, query_score = clscom.read_csv_file("vehicle_type_prediction/030519_query_type_0.92.csv")
    _, test_id, test_score = clscom.read_csv_file("vehicle_type_prediction/030519_test_type_0.92.csv")
    query_bus = clscom.apply_threshold(query_score, 3, 0.00)
    
    test_full_emb, test_full_id, test_full_track = load_emb_h5(TESTFULL['out_h5'])
    test_full_uni, tefuid = np.unique(test_full_track,  return_index=True)
    test_bus = np.array([clscom.get_majority(test_full_track,x,test_score) for x in test_full_uni])
    test_bus  = clscom.apply_threshold(test_bus, 3, 0.00)
    return query_bus, test_bus 

def classify_2_type_bus(dist_to_train):
    type1_instance = 458
    type1_thresh = 9.0
    dist_to_train_instance = np.sort(dist_to_train, axis = 1)[:,0]
    train_ids, train_paths, train_track = read_csv_dataset(TRAIN_BUS['csv_in'])
    train_uni, truid = np.unique(train_track, return_index=True)
    train_uni_id = train_ids[np.array(truid)]
    label_of_train_instance = train_uni_id[np.argsort(dist_to_train,axis = 1)][:,0]
    bus_type_label = np.array(np.logical_and(label_of_train_instance == type1_instance, dist_to_train_instance < type1_thresh))
    return (bus_type_label.astype(np.int32))

def exec_query_bus(test_train_bus_dist, querytestfull, test_bus, query_bus):
    bus2type = classify_2_type_bus(test_train_bus_dist[test_bus])
    bus_type_mask = bus2type[np.argsort(querytestfull[query_bus,:][:,test_bus])]
    bus_test_rank = test_bus[np.argsort(querytestfull[query_bus,:][:,test_bus])]
    bus_type_mask = bus_type_mask == bus_type_mask[:,0][:,None]
    bus_res = []
    for i in range(len(query_bus)):
        bus_res.append(bus_test_rank[i][bus_type_mask[i]])
    return bus_res


def select_top(inp_arr, selected):
    for i in np.argsort(inp_arr):
        if (i not in selected):
            selected.append(i)
            return selected, i

def select_NN(top1_view, tracklet_dists, k = 4):
    selected = [top1_view]
    a = top1_view
    for i in range(k):
        selected, b = select_top(tracklet_dists[a], selected)
        a = b
    return selected 

def re_ranking(top_list, scores_test):
    tmp_arr = scores_test.copy()
    tmp = np.max(tmp_arr)
    for i, track_id in enumerate(top_list):
        tmp_arr[track_id] = tmp + 1.0 - i * 0.05
    return tmp_arr

def read_csv_view(file_in):
    img_view_score = []
    with open(file_in) as fi:
        csv_reader = csv.reader(fi, delimiter = ',')
        for info in csv_reader:
            img_view_score.append(info[1].split(' '))
    return np.array(img_view_score).astype(np.float32)


def read_bow_res(file_in):
    que_res = {}
    que_conf = {}
    with open(file_in, "r") as fi:
        for line in fi:
            info = line.strip().split(",")
            k = int(info[0])
            que_res[k] = []
            que_conf[k] = []
            if info[1] == "-1":
                continue
            for img in info[2:-1]:
                data = img.split(" ")
                que_res[k].append(data[0])
                que_conf[k].append(int(data[1]))
    return que_res, que_conf

def file_name_to_id(img_path):
    return int(osp.basename(img_path).split('.')[0])-1

def top_similar_view(query_id, track_id, testqueview, test_full_track):
    same_track_let = test_full_track == track_id
    imgs_in_tracklet = np.squeeze(np.argwhere(same_track_let))
    piority = np.argsort(testqueview[query_id][same_track_let])
    scores = np.sort(testqueview[query_id][same_track_let])
    return imgs_in_tracklet[piority[0]], scores[0]
