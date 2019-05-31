#!/usr/bin/env python3
from argparse import ArgumentParser, FileType
from importlib import import_module
from itertools import count
import os

import h5py
import json
import numpy as np
from sklearn.metrics import average_precision_score
import tensorflow as tf

import common
import loss

EXP_DIR="/home/hthieu/AICityChallenge2019/track2_experiments/260218_triplet-reid_pre-trained_resnet50_veri+small_training_set/"
query_dataset="data/track2_validate_query_v3.csv"
gallery_dataset="data/track2_validate_v3.csv"
query_embeddings=os.path.join(EXP_DIR,"track2_validate_embedding.h5")
gallery_embeddings=os.path.join(EXP_DIR,"track2_validate_query_embedding.h5")

batch_size = 256

query_pids, query_fids, query_views = common.load_dataset(query_dataset, None)
gallery_pids, gallery_fids, gallery_views = common.load_dataset(gallery_dataset, None)

with h5py.File(query_embeddings, 'r') as f_query:
    query_embs = np.array(f_query['emb'])

with h5py.File(gallery_embeddings, 'r') as f_gallery:
    gallery_embs = np.array(f_gallery['emb'])