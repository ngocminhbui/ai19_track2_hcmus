#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import os

import h5py
import json
import numpy as np
import tensorflow as tf

from aggregators import AGGREGATORS
import common
import ipdb

# Required
class Embedder:
    def __init__(self, exp_root, gpu_id, batch_size = 128, crop_augment = False, ):
        self.exp_root = exp_root
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        # The GPU id to use, usually either "0" or "1"
        os.environ["CUDA_VISIBLE_DEVICES"]= str(self.gpu_id)
        
        config = tf.ConfigProto(device_count=dict(GPU=1))
        config.gpu_options.allow_growth = True
        
        # Load the args from the original experiment.
        args_file = os.path.join(self.exp_root, 'args.json')
        if os.path.isfile(args_file):
            print('Loading args from {}.'.format(args_file))
            with open(args_file, 'r') as f:
                args_resumed = json.load(f)
            
            # A couple special-cases and sanity checks
            if (args_resumed['crop_augment']) == (crop_augment is None):
                print('WARNING: crop augmentation differs between training and '
                      'evaluation.')
        else:
            raise IOError('`args.json` could not be found in: {}'.format(args_file))

#         # Check a proper aggregator is provided if augmentation is used.
#         if args.flip_augment or args.crop_augment == 'five':
#             if args.aggregator is None:
#                 print('ERROR: Test time augmentation is performed but no aggregator'
#                       'was specified.')
#                 exit(1)
#         else:
#             if args.aggregator is not None:
#                 print('ERROR: No test time augmentation that needs aggregating is '
#                       'performed but an aggregator was specified.')
#                 exit(1)  
        # Prepare models:
        self.net_input_size = (args_resumed['net_input_height'],    
                               args_resumed['net_input_width'])
        self.pre_crop_size = (args_resumed['pre_crop_height'], 
                              args_resumed['pre_crop_width'])
        
        self.crop_augment = args_resumed['crop_augment']
        self.model = import_module('nets.' + args_resumed['model_name'])
        self.head = import_module('heads.' + args_resumed['head_name'])
        self.embedding_dim = args_resumed['embedding_dim']
       
    
    def five_crops(image, crop_size):
        """ Returns the central and four corner crops of `crop_size` from `image`. """
        image_size = tf.shape(image)[:2]
        crop_margin = tf.subtract(image_size, crop_size)
        assert_size = tf.assert_non_negative(
            crop_margin, message='Crop size must be smaller or equal to the image size.')
        with tf.control_dependencies([assert_size]):
            top_left = tf.floor_div(crop_margin, 2)
            bottom_right = tf.add(top_left, crop_size)
        center       = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        top_left     = image[:-crop_margin[0], :-crop_margin[1]]
        top_right    = image[:-crop_margin[0], crop_margin[1]:]
        bottom_left  = image[crop_margin[0]:, :-crop_margin[1]]
        bottom_right = image[crop_margin[0]:, crop_margin[1]:]
        return center, top_left, top_right, bottom_left, bottom_right
    def flip_augment(image, fid, pid):
        """ Returns both the original and the horizontal flip of an image. """
        images = tf.stack([image, tf.reverse(image, [1])])
        return images, tf.stack([fid]*2), tf.stack([pid]*2)
    
    def embed_images(self, images):
        endpoints, body_prefix = self.model.endpoints(images, is_training=False, n_pids = None)
        with tf.name_scope('head'):
            endpoints = self.head.head(endpoints, self.embedding_dim,       is_training=False)
        with tf.Session() as sess:
            n_imgs = images.shape[0]
            print(">>>>>> Total images: ", n_imgs)
            print(">>>>>> Restoring from checkpoint")
            # Initialize the network/load the checkpoint.
            checkpoint = tf.train.latest_checkpoint(self.exp_root)
            print('Restoring from checkpoint: {}'.format(checkpoint))
            tf.train.Saver().restore(sess, checkpoint)
            
            modifiers = ['original']
            # Go ahead and embed the whole dataset, with all augmented versions too.
            emb_storage = np.zeros(
                (images.shape[0] * len(modifiers), self.embedding_dim), np.float32)
            emb = 0
            emb = sess.run(endpoints['emb'])
#             for start_idx in count(step=self.batch_size):
#                 try:
#                     emb = sess.run(endpoints['emb'])
#                     print('\rEmbedded batch {}-{}/{}'.format(
#                             start_idx, start_idx + len(emb), len(emb_storage)),
#                         flush=True, end='')
# #                     emb_storage[start_idx:start_idx + len(emb)] = emb
#                 except tf.errors.OutOfRangeError:
#                     break  # This just indicates the end of the dataset.
        tf.reset_default_graph()
        return emb
    
    def embed_csv_file(self, 
                        image_root, 
                        csv_file,
                        emb_file,
                        loading_threads = 8,
                        flip = False,
                        crop = False,
                        aggregator = 'mean'):
        data_ids, data_fids, data_fols = common.load_dataset(csv_file, image_root)
        data_ids = data_ids.astype(np.int32)
        data_fols = data_fols.astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices(data_fids)
        dataset = dataset.map(
        lambda fid: common.fid_to_image(
            fid, tf.constant('dummy'), image_root=image_root,
            image_size=self.pre_crop_size if self.crop_augment else self.net_input_size),
        num_parallel_calls=loading_threads)
        
        # Augment the data if specified by the arguments.
        # `modifiers` is a list of strings that keeps track of which augmentations
        # have been applied, so that a human can understand it later on.
        modifiers = ['original']
        if flip:
            dataset = dataset.map(Embedder.flip_augment)
            dataset = dataset.apply(tf.contrib.data.unbatch())
            modifiers = [o + m for m in ['', '_flip'] for o in modifiers]
        print(flip, crop)
        if crop == 'center':
            dataset = dataset.map(lambda im, fid, pid:
                (five_crops(im, net_input_size)[0], fid, pid))
            modifiers = [o + '_center' for o in modifiers]
        elif crop == 'five':
            dataset = dataset.map(lambda im, fid, pid: (
                tf.stack(Embedder.five_crops(im, self.net_input_size)),
                tf.stack([fid]*5),
                tf.stack([pid]*5)))
            dataset = dataset.apply(tf.contrib.data.unbatch())
            modifiers = [o + m for o in modifiers for m in [
                '_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]
        elif crop == 'avgpool':
            modifiers = [o + '_avgpool' for o in modifiers]
        else:
            modifiers = [o + '_resize' for o in modifiers]

        # Group it back into PK batches.
        dataset = dataset.batch(self.batch_size)

        # Overlap producing and consuming.
        dataset = dataset.prefetch(1)

        images, _, _ = dataset.make_one_shot_iterator().get_next()


        endpoints, body_prefix = self.model.endpoints(images, is_training=False)
        with tf.name_scope('head'):
            endpoints = self.head.head(endpoints, self.embedding_dim, is_training=False)
        
#         emb_file = os.path.join(self.exp_root, emb_file)
        print("Save h5 file to: ", emb_file)
        with h5py.File(emb_file, 'w') as f_out, tf.Session() as sess:
            # Initialize the network/load the checkpoint.
            checkpoint = tf.train.latest_checkpoint(self.exp_root)
            print('Restoring from checkpoint: {}'.format(checkpoint))
            tf.train.Saver().restore(sess, checkpoint)

            # Go ahead and embed the whole dataset, with all augmented versions too.
            emb_storage = np.zeros(
                (len(data_fids) * len(modifiers), self.embedding_dim), np.float32)
            for start_idx in count(step=self.batch_size):
                try:
                    emb = sess.run(endpoints['emb'])
                    print('\rEmbedded batch {}-{}/{}'.format(
                            start_idx, start_idx + len(emb), len(emb_storage)),
                        flush=True, end='')
                    emb_storage[start_idx:start_idx + len(emb)] = emb
                except tf.errors.OutOfRangeError:
                    break  # This just indicates the end of the dataset.

            print()
            print("Done with embedding, aggregating augmentations...", flush=True)
            print(emb_storage.shape)
            if len(modifiers) > 1:
                # Pull out the augmentations into a separate first dimension.
                emb_storage = emb_storage.reshape(len(data_fids), len(modifiers), -1)
                emb_storage = emb_storage.transpose((1,0,2))  # (Aug,FID,128D)

                # Store the embedding of all individual variants too.
                emb_dataset = f_out.create_dataset('emb_aug', data=emb_storage)

                # Aggregate according to the specified parameter.
                emb_storage = AGGREGATORS[aggregator](emb_storage)
            print(emb_storage.shape)
            # Store the final embeddings.
            f_out.create_dataset('emb', data=emb_storage)
            f_out.create_dataset('id', data=data_ids)
            f_out.create_dataset('fol_id', data=data_fols)

            # Store information about the produced augmentation and in case no crop
            # augmentation was used, if the images are resized or avg pooled.
            f_out.create_dataset('augmentation_types', data=np.asarray(modifiers, dtype='|S'))
        
        tf.reset_default_graph()
    
    def do_experiment_with_config(self,config):
        self.embed_csv_file(
            config.QUE_IMG_ROOT,
            config.QUE_FILE,
            config.QUE_EMB_FILE,
            flip = config.FLIP,
            crop = config.CROP
        )
        
        self.embed_csv_file(
            config.GAL_IMG_ROOT,
            config.GAL_FILE,
            config.GAL_EMB_FILE,
            flip = config.FLIP,
            crop = config.CROP
        )
            
        
# # Optional

# parser.add_argument(
#     '--filename', default=None,
#     help='Name of the HDF5 file in which to store the embeddings, relative to'
#          ' the `experiment_root` location. If omitted, appends `_embeddings.h5`'
#          ' to the dataset name.')

# parser.add_argument(
#     '--flip_augment', action='store_true', default=False,
#     help='When this flag is provided, flip augmentation is performed.')

# parser.add_argument(
#     '--crop_augment', choices=['center', 'avgpool', 'five'], default=None,
#     help='When this flag is provided, crop augmentation is performed.'
#          '`avgpool` means the full image at the precrop size is used and '
#          'the augmentation is performed by the average pooling. `center` means'
#          'only the center crop is used and `five` means the four corner and '
#          'center crops are used. When not provided, by default the image is '
#          'resized to network input size.')

# parser.add_argument(
#     '--aggregator', choices=AGGREGATORS.keys(), default=None,
#     help='The type of aggregation used to combine the different embeddings '
#          'after augmentation.')

# parser.add_argument(
#     '--quiet', action='store_true', default=False,
#     help='Don\'t be so verbose.')

    


#     # Possibly auto-generate the output filename.
#     if args.filename is None:
#         basename = os.path.basename(args.dataset)
#         args.filename = os.path.splitext(basename)[0] + '_embeddings.h5'
#     args.filename = os.path.join(args.experiment_root, args.filename)

###############################