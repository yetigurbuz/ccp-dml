import tensorflow as tf

import numpy as np

import os

from six.moves import cPickle

import cv2

from .dataset_base import Dataset

class ImagenetML(Dataset):
    '''@inproceedings{Ravi2017OptimizationAA,
          title={Optimization as a Model for Few-Shot Learning},
          author={Sachin Ravi and H. Larochelle},
          booktitle={ICLR},
          year={2017}
        }'''

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        example_names = {}
        num_classes = {}
        for s in ['train', 'val', 'test']:

            data_folder = os.path.join(base_dir, 'miniimagenet', 'data')
            subset_splits = os.path.join(base_dir, 'miniimagenet', 'splits', 'ravi-larochelle', '{}.txt'.format(s))
            with open(subset_splits, mode='r') as f:
                class_ids = [n.strip() for n in f.readlines()]

            subset = s if s != 'test' else 'eval'
            example_names[subset] = []
            num_classes[subset] = len(class_ids)
            for k, cate_id in enumerate(class_ids):
                cate_folder = os.path.join(data_folder, cate_id)
                _ = [example_names[subset].append('{} {} 84 84'.format(k, os.path.join(data_folder, cate_folder, fname)))
                     for fname in [x for x in os.listdir(cate_folder)
                                   if x.endswith('.jpg')]]

        train_images = np.stack(
            [cv2.imread(eg.split(' ')[1], cv2.IMREAD_UNCHANGED)
             for eg in example_names['train']],
            axis=0)
        train_images = train_images.astype(np.float32) / 255.

        dataset_mean = np.mean(train_images, axis=0)[:, :, ::-1].tolist()
        dataset_std = np.std(train_images, axis=0)[:, :, ::-1].tolist()

        shard_size = 10000

        return example_names, num_classes, shard_size, dataset_mean, dataset_std

    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        label, image_path, h, w = example_name.split(' ')

        arg_list = [{'image_path': image_path, 'label': int(label), 'height': int(h), 'width': int(w)}]

        return arg_list

class ImagenetMini(Dataset):
    '''@inproceedings{Ravi2017OptimizationAA,
          title={Optimization as a Model for Few-Shot Learning},
          author={Sachin Ravi and H. Larochelle},
          booktitle={ICLR},
          year={2017}
        }

        '''

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original datasets organization
        main_folder = os.path.join(base_dir, '..', 'ImagenetML', 'miniimagenet')

        example_names = {}
        num_classes = {}
        class_ids = []
        for s in ['train', 'val', 'test']:

            data_folder = os.path.join(main_folder, 'data')
            subset_splits = os.path.join(main_folder, 'splits', 'ravi-larochelle', '{}.txt'.format(s))
            with open(subset_splits, mode='r') as f:
                _ = [class_ids.append(n.strip()) for n in f.readlines()]

            subset = s if s != 'test' else 'eval'
            example_names[subset] = []

        total_classes = len(class_ids)
        num_classes['train'] = total_classes
        num_classes['val'] = total_classes
        num_classes['eval'] = total_classes

        train_ratio = 0.65
        val_ratio = 0.15
        eval_ratio = 0.20

        for k, cate_id in enumerate(class_ids):
            cate_folder = os.path.join(data_folder, cate_id)
            class_samples = ['{} {} 84 84'.format(k, os.path.join(data_folder, cate_folder, fname))
                 for fname in [x for x in os.listdir(cate_folder)
                               if x.endswith('.jpg')]]

            num_samples = len(class_samples)
            t1 = round(train_ratio * num_samples)
            t3 = round(eval_ratio * num_samples)
            t2 = num_samples - (t1 + t3)

            train_set = class_samples[:t1]
            val_set = class_samples[t1:t1 + t2]
            eval_set = class_samples[t1 + t2:]

            [example_names['train'].append(e) for e in train_set]
            [example_names['val'].append(e) for e in val_set]
            [example_names['eval'].append(e) for e in eval_set]


        train_images = np.stack(
            [cv2.imread(eg.split(' ')[1], cv2.IMREAD_UNCHANGED)
             for eg in example_names['train']],
            axis=0)
        train_images = train_images.astype(np.float32) / 255.

        dataset_mean = np.mean(train_images, axis=0)[:, :, ::-1].tolist()
        dataset_std = np.std(train_images, axis=0)[:, :, ::-1].tolist()

        shard_size = 10000

        return example_names, num_classes, shard_size, dataset_mean, dataset_std

    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        label, image_path, h, w = example_name.split(' ')

        arg_list = [{'image_path': image_path, 'label': int(label), 'height': int(h), 'width': int(w)}]

        return arg_list