import tensorflow as tf

import numpy as np

import os

from six.moves import cPickle

import cv2


from .dataset_base import Dataset

class Cifar100(Dataset):


    def _reOrganizeDataset(self, base_dir):

        def saveImages(target_path, flattened_data, labels, batch_id):

            file_names = []

            num_data = len(labels)
            prog_bar = tf.keras.utils.Progbar(target=num_data)
            for k in range(num_data):

                '''if k % 1000 == 0:
                    print('INFO:Dataset:{}:saving ({}) / ({})'.format('images', k + 1, num_data))'''

                img_name = os.path.join(target_path,
                                        'c%03d' % (labels[k] + 1),
                                        '%03d_%05d_%03d.png' % (batch_id + 1, k + 1, labels[k] + 1))

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
        num_classes = {'train': 100, 'val': 100, 'eval': 100}
        raw_data_dir = os.path.join(base_dir, 'cifar-100-python')
        print(self.information('Downloading and extracting the datasets...'))
        # download and extract datasets if not exists
        if not os.path.exists(raw_data_dir):
            download_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            fname = os.path.split(download_url)[-1]

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        batch_names = {}
        batch_names['train'] = [os.path.join(raw_data_dir, 'train')]
        batch_names['val'] = [os.path.join(raw_data_dir, 'test')]
        batch_names['eval'] = [os.path.join(raw_data_dir, 'test')]

        # datasets with raw images

        # create related folders
        path_to_train_images = os.path.join(base_dir, 'images', 'training')
        if not tf.io.gfile.exists(path_to_train_images):
            tf.io.gfile.makedirs(path_to_train_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_train_images, 'c%03d' % (i + 1))) for i in range(num_classes['train'])]
            print(self.information('Created class folders for %s'%path_to_train_images))

        path_to_validation_images = os.path.join(base_dir, 'images', 'validation')
        if not tf.io.gfile.exists(path_to_validation_images):
            tf.io.gfile.makedirs(path_to_validation_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_validation_images, 'c%03d' % (i + 1))) for i in
             range(num_classes['val'])]
            print(self.information('Created class folders for %s' % path_to_validation_images))

        path_to_test_images = os.path.join(base_dir, 'images', 'evaluation')
        if not tf.io.gfile.exists(path_to_test_images):
            tf.io.gfile.makedirs(path_to_test_images)
            [tf.io.gfile.makedirs(os.path.join(path_to_test_images, 'c%03d' % (i + 1))) for i in range(num_classes['eval'])]
            print(self.information('Created class folders for %s' % path_to_test_images))

        # training
        with tf.io.gfile.GFile(batch_names['train'][0], 'rb') as f:
            data_dict = cPickle.load(f, encoding='bytes')
            train_data = data_dict[b'data']
            train_labels = data_dict[b'fine_labels']

        # validation
        with tf.io.gfile.GFile(batch_names['val'][0], 'rb') as f:
            data_dict = cPickle.load(f, encoding='bytes')
            val_data = data_dict[b'data']
            val_labels = data_dict[b'fine_labels']

        # test
        with tf.io.gfile.GFile(batch_names['eval'][0], 'rb') as f:
            data_dict = cPickle.load(f, encoding='bytes')
            test_data = data_dict[b'data']
            test_labels = data_dict[b'fine_labels']

        # compute datasets mean to be used in preprocessing later
        dataset_mean = np.mean(train_data.astype(np.float32) / 255., axis=0)
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

        classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                   ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                   ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
                   ['bottles', 'bowls', 'cans', 'cups', 'plates'],
                   ['apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'],
                   ['clock', 'computer keyboard', 'lamp', 'telephone', 'television'],
                   ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                   ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                   ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                   ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                   ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                   ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                   ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                   ['crab', 'lobster', 'snail', 'spider', 'worm'],
                   ['baby', 'boy', 'girl', 'man', 'woman'],
                   ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                   ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                   ['maple', 'oak', 'palm', 'pine', 'willow'],
                   ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'],
                   ['lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']]
        super_classes = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
                         'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
                         'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
                         'large_omnivores_and_herbivores',
                         'medium-sized_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
                         'trees', 'vehicles_1', 'vehicles_2']

        sub_classes = ['apples', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
                       'beetle', 'bicycle', 'bottles', 'bowls', 'boy', 'bridge', 'bus',
                       'butterfly', 'camel', 'cans', 'castle', 'caterpillar', 'cattle',
                       'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
                       'crocodile', 'cups', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
                       'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn-mower',
                       'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple',
                       'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges',
                       'orchids', 'otter', 'palm', 'pears', 'pickup_truck', 'pine',
                       'plain', 'plates', 'poppies', 'porcupine', 'possum', 'rabbit',
                       'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 'seal',
                       'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                       'spider', 'squirrel', 'streetcar', 'sunflowers', 'sweet_peppers',
                       'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                       'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale',
                       'willow', 'wolf', 'woman', 'worm']

        super_name_to_label = {s: int(l) for l, s in enumerate(super_classes)}
        sub_name_to_label = {s: int(l) for l, s in enumerate(sub_classes)}

        super_label_to_name = {int(v): k for k, v in super_name_to_label.items()}
        sub_label_to_name = {int(v): k for k, v in sub_name_to_label.items()}

        return {'total': 100,
                'super_name_to_label': super_name_to_label, 'super_label_to_name': super_label_to_name,
                'sub_name_to_label': sub_name_to_label, 'sub_label_to_name': sub_label_to_name}

    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        image_path = example_name
        label = int(image_path.split('_')[-1].split('.')[0])

        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image)
        image_shape = tf.shape(image).numpy()
        height = image_shape[0]
        width = image_shape[1]

        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width}]

        return arg_list

class Cifar100ML(Dataset):
    '''@inproceedings{
        bertinetto2018metalearning,
        title={Meta-learning with differentiable closed-form solvers},
        author={Luca Bertinetto and Joao F. Henriques and Philip Torr and Andrea Vedaldi},
        booktitle={International Conference on Learning Representations},
        year={2019},
        url={https://openreview.net/forum?id=HyxnZh0ct7},
        }'''

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        example_names = {}
        num_classes = {}
        for s in ['train', 'val', 'test']:
            data_folder = os.path.join(base_dir, 'cifar100', 'data')
            subset_splits = os.path.join(base_dir, 'cifar100', 'splits', 'bertinetto', '{}.txt'.format(s))
            with open(subset_splits, mode='r') as f:
                class_ids = [n.strip() for n in f.readlines()]

            subset = s if s != 'test' else 'eval'
            example_names[subset] = []
            num_classes[subset] = len(class_ids)
            for k, cate_id in enumerate(class_ids):
                cate_folder = os.path.join(data_folder, cate_id)
                _ = [example_names[subset].append(
                    '{} {} 32 32'.format(k, os.path.join(data_folder, cate_folder, fname)))
                     for fname in [x for x in os.listdir(cate_folder)
                                   if x.endswith('.png')]]

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
