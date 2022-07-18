import tensorflow as tf

import numpy as np

import os

from six.moves import cPickle

from .dataset_base import Dataset

import PIL

class Cifar10(Dataset):

    def _reOrganizeDataset(self, base_dir):

        def saveImages(target_path, flattened_data, labels, batch_id):

            file_names = []

            num_data = len(labels)
            prog_bar = tf.keras.utils.Progbar(target=num_data)
            for k in range(num_data):

                '''if k % 1000 == 0:
                    print('INFO:Dataset:{}:saving ({}) / ({})'.format('images', k + 1, num_data))'''


                img_name = os.path.join(target_path,
                                        'c%02d'%(labels[k] + 1),
                                        '%02d_%05d_%02d.png'%(batch_id + 1, k + 1, labels[k] + 1))

                with tf.device('CPU:0'):
                    if not tf.io.gfile.exists(img_name):
                        img = np.transpose(np.reshape(flattened_data[k], (3, 32, 32)), (1, 2, 0))
                        img_decoded = tf.image.encode_png(img)
                        with tf.io.gfile.GFile(img_name, 'bw') as f:
                            f.write(img_decoded.numpy())

                file_names.append(img_name)
                prog_bar.add(1)

            return file_names

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original datasets organization
        num_classes = {'train': 10, 'val': 10, 'eval': 10}
        raw_data_dir = os.path.join(base_dir, 'cifar-10-batches-py')
        print(self.information('Downloading and extracting the datasets...'))
        # download and extract datasets if not exists
        if not os.path.exists(raw_data_dir):
            download_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            fname = os.path.split(download_url)[-1]

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        batch_names = {}
        batch_names['train'] = [os.path.join(raw_data_dir, 'data_batch_%d' % i) for i in range(1, 5)]
        batch_names['val'] = [os.path.join(raw_data_dir, 'data_batch_5')]
        batch_names['eval'] = [os.path.join(raw_data_dir, 'test_batch')]

        # datasets with raw images

        # create related folders
        path_to_train_images = os.path.join(base_dir, 'images', 'training')
        if not tf.io.gfile.exists(path_to_train_images):
            tf.io.gfile.makedirs(path_to_train_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_train_images, 'c%02d'%(i + 1))) for i in range(num_classes['train'])]
            print(self.information('Created class folders for %s'%path_to_train_images))

        path_to_validation_images = os.path.join(base_dir, 'images', 'validation')
        if not tf.io.gfile.exists(path_to_validation_images):
            tf.io.gfile.makedirs(path_to_validation_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_validation_images, 'c%02d'%(i + 1))) for i in range(num_classes['val'])]
            print(self.information('Created class folders for %s' % path_to_validation_images))

        path_to_test_images = os.path.join(base_dir, 'images', 'evaluation')
        if not tf.io.gfile.exists(path_to_test_images):
            tf.io.gfile.makedirs(path_to_test_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_test_images, 'c%02d'%(i + 1))) for i in range(num_classes['eval'])]
            print(self.information('Created class folders for %s' % path_to_test_images))

        # training
        train_data_list = []
        train_labels = []
        for batch_name in batch_names['train']:
            with tf.io.gfile.GFile(batch_name, 'rb') as f:
                data_dict = cPickle.load(f, encoding='bytes')
                train_data_list.append(data_dict[b'data'])
                train_labels += data_dict[b'labels']

        train_data = np.concatenate(train_data_list, axis=0)

        # validation
        with tf.io.gfile.GFile(batch_names['val'][0], 'rb') as f:
            data_dict = cPickle.load(f, encoding='bytes')
            val_data = data_dict[b'data']
            val_labels = data_dict[b'labels']

        # test
        with tf.io.gfile.GFile(batch_names['eval'][0], 'rb') as f:
            data_dict = cPickle.load(f, encoding='bytes')
            test_data = data_dict[b'data']
            test_labels = data_dict[b'labels']

        # compute datasets mean to be used in preprocessing later
        dataset_mean = np.mean(np.concatenate((train_data, val_data), axis=0).astype(np.float32) / 255., axis=0)
        dataset_mean = np.transpose(np.reshape(dataset_mean, (3, 32, 32)), (1, 2, 0))

        # now write raw images
        example_names = {}

        print(self.information('Writing training images'))
        example_names['train'] = saveImages(path_to_train_images, train_data, train_labels, 0)

        print(self.information('Writing validation images'))
        example_names['val'] = saveImages(path_to_validation_images, val_data, val_labels, 1)

        print(self.information('Writing evaluation images'))
        example_names['eval'] = saveImages(path_to_test_images, test_data, test_labels, 2)

        shard_size = 10000

        return example_names, num_classes, shard_size, dataset_mean.tolist(), 1.

    def _classInfo(self):
        name_to_label = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9,
        }
        label_to_name = {int(v): k for k, v in name_to_label.items()}

        return {'total': 10, 'name_to_label': name_to_label, 'label_to_name': label_to_name}

    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        image_path = example_name
        label = int(image_path.split('_')[-1].split('.')[0])

        image = PIL.Image.open(image_path, 'r')
        height = image.height
        width = image.width

        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width}]

        return arg_list

class Cifar10ML(Dataset):

    def _reOrganizeDataset(self, base_dir):

        def saveImages(target_path, flattened_data, labels, batch_id, label_map):

            file_names = []

            num_data = len(labels)
            prog_bar = tf.keras.utils.Progbar(target=num_data)
            for k in range(num_data):

                '''if k % 1000 == 0:
                    print('INFO:Dataset:{}:saving ({}) / ({})'.format('images', k + 1, num_data))'''

                label = label_map[labels[k]]
                img_name = os.path.join(target_path,
                                        'c%02d'%(label),
                                        '%02d_%05d_%02d.png'%(batch_id + 1, k + 1, label))

                with tf.device('CPU:0'):
                    if not tf.io.gfile.exists(img_name):
                        img = np.transpose(np.reshape(flattened_data[k], (3, 32, 32)), (1, 2, 0))
                        img_decoded = tf.image.encode_png(img)
                        with tf.io.gfile.GFile(img_name, 'bw') as f:
                            f.write(img_decoded.numpy())

                file_names.append(img_name)
                prog_bar.add(1)

            return file_names


        # original datasets organization
        num_classes = {'train': 5, 'val': 5, 'eval': 5}
        raw_data_dir = os.path.join(base_dir, 'cifar-10-batches-py')
        print(self.information('Downloading and extracting the datasets...'))
        # download and extract datasets if not exists
        if not os.path.exists(raw_data_dir):
            download_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            fname = os.path.split(download_url)[-1]

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        batch_names = {}
        batch_names['train'] = [os.path.join(raw_data_dir, 'data_batch_%d' % i) for i in range(1, 5)]
        batch_names['val'] = [os.path.join(raw_data_dir, 'data_batch_5')]
        batch_names['eval'] = [os.path.join(raw_data_dir, 'test_batch')]

        # datasets with raw images

        # create related folders
        path_to_train_images = os.path.join(base_dir, 'images', 'training')
        if not tf.io.gfile.exists(path_to_train_images):
            tf.io.gfile.makedirs(path_to_train_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_train_images, 'c%02d'%(i + 1))) for i in range(num_classes['train'])]
            print(self.information('Created class folders for %s'%path_to_train_images))

        path_to_validation_images = os.path.join(base_dir, 'images', 'validation')
        if not tf.io.gfile.exists(path_to_validation_images):
            tf.io.gfile.makedirs(path_to_validation_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_validation_images, 'c%02d'%(i + 1))) for i in range(num_classes['val'])]
            print(self.information('Created class folders for %s' % path_to_validation_images))

        path_to_test_images = os.path.join(base_dir, 'images', 'evaluation')
        if not tf.io.gfile.exists(path_to_test_images):
            tf.io.gfile.makedirs(path_to_test_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_test_images, 'c%02d'%(i + 1))) for i in range(num_classes['eval'])]
            print(self.information('Created class folders for %s' % path_to_test_images))

        # training
        train_data_list = []
        train_labels = []
        for batch_name in batch_names['train']:
            with tf.io.gfile.GFile(batch_name, 'rb') as f:
                data_dict = cPickle.load(f, encoding='bytes')
                train_data_list.append(data_dict[b'data'])
                train_labels += data_dict[b'labels']

        train_data = np.concatenate(train_data_list, axis=0)

        # validation
        with tf.io.gfile.GFile(batch_names['val'][0], 'rb') as f:
            data_dict = cPickle.load(f, encoding='bytes')
            val_data = data_dict[b'data']
            val_labels = data_dict[b'labels']

        # test
        with tf.io.gfile.GFile(batch_names['eval'][0], 'rb') as f:
            data_dict = cPickle.load(f, encoding='bytes')
            test_data = data_dict[b'data']
            test_labels = data_dict[b'labels']

        # splits for zero shot setting
        train_label_subset = [0, 1, 2, 3, 4]
        test_label_subset = [5, 6, 7, 8, 9]
        train_label_map = {k: v + 1 for v, k in enumerate(train_label_subset)}
        test_label_map = {k: v + 1 for v, k in enumerate(test_label_subset)}

        all_data = np.concatenate([train_data, val_data, test_data], axis=0)
        all_labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)

        split_ids = [[], []]

        [split_ids[int(lbl in test_label_subset)].append(id) for id, lbl in enumerate(all_labels)]

        train_ids, test_ids = split_ids
        train_data = all_data[train_ids]
        train_labels = all_labels[train_ids]
        test_data = all_data[test_ids]
        test_labels = all_labels[test_ids]

        # compute datasets mean to be used in preprocessing later
        dataset_mean = np.mean(train_data.astype(np.float32) / 255., axis=0)
        dataset_mean = np.transpose(np.reshape(dataset_mean, (3, 32, 32)), (1, 2, 0))

        # now write raw images
        example_names = {}

        print(self.information('Writing training images'))
        example_names['train'] = saveImages(path_to_train_images, train_data, train_labels, 0, train_label_map)

        print(self.information('Writing validation images'))
        example_names['val'] = example_names['train'] #saveImages(path_to_validation_images, val_data, val_labels, 1)

        print(self.information('Writing evaluation images'))
        example_names['eval'] = saveImages(path_to_test_images, test_data, test_labels, 2, test_label_map)

        shard_size = 10000

        return example_names, num_classes, shard_size, dataset_mean.tolist(), 1.

    def _classInfo(self):
        name_to_label = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9,
        }
        label_to_name = {int(v): k for k, v in name_to_label.items()}

        return {'total': 10, 'name_to_label': name_to_label, 'label_to_name': label_to_name}

    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        image_path = example_name
        label = int(image_path.split('_')[-1].split('.')[0])

        image = PIL.Image.open(image_path, 'r')
        height = image.height
        width = image.width

        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width}]

        return arg_list

