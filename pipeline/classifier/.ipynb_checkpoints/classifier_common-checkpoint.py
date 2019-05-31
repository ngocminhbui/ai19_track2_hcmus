import csv
import numpy as np
import h5py
from matplotlib import pyplot as plt

def read_csv_file(file_name):
    img_id = []
    img_view = []
    img_scores = []
    with open(file_name) as fi:
        csv_reader = csv.reader(fi, delimiter = ',')
        for info in csv_reader:
            if (info[0]==""):
                continue
            img_id.append(info[0])
            img_view.append(int(info[1]))
            img_scores.append(info[2].split(" "))
    img_id = np.array(img_id)
    img_view = np.array(img_view, dtype = np.int32)
    img_scores = np.array(img_scores, dtype = np.float32)
    return img_id, img_view, img_scores

def get_by_view(arr, idx, arr_val):
    arr_idx = np.squeeze(np.argwhere(arr==idx))
    return arr_val[list(arr_idx)]

def read_h5_file(filename, config):
    f = h5py.File(filename, 'r')
    fids = np.array(f['img_id'])
    scores = np.array(f[config.INFO_ID])
    preds = np.argmax(scores,axis=1)
    confs = np.max(scores,axis=1)
#     embs = np.array(f['emb'])
    f.close()
    return fids, scores, preds, confs #, embs

def show_img_list(img_list):
    n = 100 if len(img_list) - 1 > 100 else len(img_list) -1
    w=10
    h=10
    fig=plt.figure(figsize=(16, 16))
    fig.tight_layout()
    columns = 10
    rows = 10
    for i in range(1, n +1):
        img = plt.imread(img_list[i])
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    
def get_majority(arr, idx, arr_val):
    scores = get_by_view(arr,idx,arr_val)
    res = np.sum(scores,axis=0)
#     res = np.argmax(res)
    return(res)

def select_cls(cls_id):
    print(config.ID2NAME[cls_id])
    tmp = np.ones(preds.shape, dtype=np.int32) * cls_id
    return np.where(tmp==preds)

def apply_threshold(scores, label, thresh):
    label_raw = np.argmax(scores, axis = 1)
    mask = np.max(scores,axis = 1) >= thresh
    return np.where(np.logical_and(label_raw == label, mask))[0]