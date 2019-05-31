import sys
sys.path.insert(0, '/home/hthieu/AICityChallenge2019/classifier')
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import utils
import argparse
from nets.model import get_model_fn
from dataloader import DataLoader
import numpy as np
import h5py
class Classifier:
    def __init__(self,model, gpu_id, batch_size = 256):
        self.gpu_id = gpu_id
        self.exp_root = model.MODEL_ROOT
        self.img_size = model.IMG_SIZE
        self.model = model.MODEL_NAME
        self.batch_size = batch_size
        self.num_readers = 8
        self.no_classes = model.NUM_CLASSES
        self.info_id = model.info_id
    
    def classifiy(self, inp_root, inp_file, out_file):
        ##################
        ## Load dataset ##
        ##################
        print(os.path.join(inp_root,inp_file))
        image_paths, images, labels, no_samples, no_classes = DataLoader.get_dataset_from_folder(
                                dataset_name = "",
                                dataset_root = inp_root, 
                                csv_file = inp_file,
                                split = "",
                                batch_size = self.batch_size,
                                shuffle = False,
                                classification = True)
        ################
        ## Load Model ##
        ################
        model_fn = get_model_fn(self.model)
        net, layers = model_fn(images, image_size = self.img_size, num_classes = self.no_classes,
                                is_training = False, scope = self.model)
        
        predicts = tf.argmax(net, axis = 1)
        confidents = tf.reduce_max(tf.nn.softmax(net), axis = 1)
        accuracy = slim.metrics.accuracy(predicts, tf.cast(labels, tf.int64))


        config = tf.ConfigProto(device_count=dict(GPU=1))
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        model_path = tf.train.latest_checkpoint(self.exp_root)
        if model_path is not None:
            saver.restore(sess, model_path)
            print("Restore model succcessfully from {}".format(model_path))
        else:
            raise ValueError("Cannot Found model from {}".format(self.exp_root))

        #############
        ## Testing ##
        #############
        timer = utils.Timer()
        num_batchs = int(no_samples / self.batch_size) + 1
        total = 0.0

        if not (os.path.exists(os.path.dirname(out_file))):
            os.makedirs(os.path.dirname(out_file))

        with h5py.File(out_file, 'w') as fo:
            total_scores =[]
            for iter in range(1, num_batchs + 1):
                timer.tic()
                img_paths, acc, preds, confs, scores =  sess.run([image_paths, accuracy, predicts, confidents, net])
                print("Iteration [{}/{}]:".format(iter, num_batchs))
                print("\t>> Accuracy:\t{}".format(acc))
                print("\t>> Executed Time:\t{} sec/iter".format(timer.toc())) 
                total_scores.append(scores)
            
            total_scores = np.concatenate(total_scores, axis = 0)
            fo.create_dataset(self.info_id, data = total_scores)
            total += acc
            fo.close()
        sess.close()
        tf.reset_default_graph()
