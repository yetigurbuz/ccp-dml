import tensorflow as tf
import PIL
import os
import numpy as np

from .dataset_base import Dataset

class CUB200_2011(Dataset):

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original datasets organization
        main_folder = os.path.join(base_dir, 'CUB_200_2011')
        image_folder = os.path.join(main_folder, 'images')

        print(self.information('Downloading and extracting the datasets...'))
        # download and extract datasets if not exists
        if not os.path.exists(main_folder):
            download_url = 'https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz'
            fname = os.path.split(download_url)[-1]

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        train_end_id = 5864

        with open(os.path.join(main_folder, 'images.txt'), 'r') as f:
            image_list = f.readlines()

        with open(os.path.join(main_folder, 'bounding_boxes.txt'), 'r') as f:
            bbox_list = f.readlines()

        example_name_list = \
            ['bbox: ' + bbox.split('\n')[0].split(' ', maxsplit=1)[-1] +
             ' imagefile: ' + os.path.join(image_folder, imgf.split('\n')[0].split(' ')[-1])
             for bbox, imgf in zip(bbox_list, image_list)]


        example_names = {}

        example_names['train'] = example_name_list[:train_end_id]

        example_names['val'] = None

        example_names['eval'] = example_name_list[train_end_id:]

        num_classes = {'train': 100, 'val': 0, 'eval': 100}

        shard_size = 1500

        dataset_mean = [[[0.485, 0.456, 0.406]]]
        dataset_std = [[[0.229, 0.224, 0.225]]]

        return example_names, num_classes, shard_size, dataset_mean, dataset_std


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        [bbox, image_path] = example_name.strip().split('imagefile: ')
        label = int(image_path.split(os.path.sep)[-2].split('.')[0])
        [xmin, ymin, b_width, b_height] = bbox.split(' ')[1:-1]

        with PIL.Image.open(image_path, 'r') as img:
            width, height = img.size

        xmin = float(xmin)
        xmax = min(xmin + float(b_width), float(width))

        ymin = float(ymin)
        ymax = min(ymin + float(b_height), float(height))


        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width,
                     'ymins': ymin/height, 'ymaxs': ymax/height,
                     'xmins': xmin/width, 'xmaxs': xmax/width}]

        return arg_list

class CUB200_2011Closed(Dataset):

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original datasets organization
        main_folder = os.path.join(base_dir, '..', 'CUB200_2011', 'CUB_200_2011')
        image_folder = os.path.join(main_folder, 'images')

        train_end_id = 5864

        with open(os.path.join(main_folder, 'images.txt'), 'r') as f:
            image_list = f.readlines()

        with open(os.path.join(main_folder, 'bounding_boxes.txt'), 'r') as f:
            bbox_list = f.readlines()

        example_name_list = \
            ['bbox: ' + bbox.split('\n')[0].split(' ', maxsplit=1)[-1] +
             ' imagefile: ' + os.path.join(image_folder, imgf.split('\n')[0].split(' ')[-1])
             for bbox, imgf in zip(bbox_list, image_list)]

        class_list = [[] for _ in range(200)]
        [
            class_list[int(e.strip().split(
                'imagefile: ')[-1].split(os.path.sep)[-2].split('.')[0]) - 1].append(e)
            for e in example_name_list]

        train_ratio = 0.65
        val_ratio = 0.15
        eval_ratio = 0.20

        example_names = {}

        example_names['train'] = []

        example_names['val'] = []

        example_names['eval'] = []

        for c in class_list:
            num_samples = len(c)
            t1 = round(train_ratio * num_samples)
            t3 = round(eval_ratio * num_samples)
            t2 = num_samples - (t1 + t3)

            train_set = c[:t1]
            val_set = c[t1:t1+t2]
            eval_set = c[t1+t2:]

            [example_names['train'].append(e) for e in train_set]
            [example_names['val'].append(e) for e in val_set]
            [example_names['eval'].append(e) for e in eval_set]


        num_classes = {'train': 200, 'val': 200, 'eval': 200}

        shard_size = 1500

        dataset_mean = [[[0.485, 0.456, 0.406]]]
        dataset_std = [[[0.229, 0.224, 0.225]]]

        return example_names, num_classes, shard_size, dataset_mean, dataset_std


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        [bbox, image_path] = example_name.strip().split('imagefile: ')
        label = int(image_path.split(os.path.sep)[-2].split('.')[0])
        [xmin, ymin, b_width, b_height] = bbox.split(' ')[1:-1]

        with PIL.Image.open(image_path, 'r') as img:
            width, height = img.size

        xmin = float(xmin)
        xmax = min(xmin + float(b_width), float(width))

        ymin = float(ymin)
        ymax = min(ymin + float(b_height), float(height))


        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width,
                     'ymins': ymin/height, 'ymaxs': ymax/height,
                     'xmins': xmin/width, 'xmaxs': xmax/width}]

        return arg_list

class CUB200_2011Noisy(Dataset):

    def __init__(self, **kwargs):
        super(CUB200_2011Noisy, self).__init__(**kwargs)

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original datasets organization
        main_folder = os.path.join(base_dir, 'CUB_200_2011')
        image_folder = os.path.join(main_folder, 'images')

        print(self.information('Downloading and extracting the datasets...'))
        # download and extract datasets if not exists
        if not os.path.exists(main_folder):
            download_url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
            fname = os.path.split(download_url)[-1]

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        train_end_id = 5864
        num_train_classes = 100

        with open(os.path.join(main_folder, 'images.txt'), 'r') as f:
            image_list = f.readlines()

        with open(os.path.join(main_folder, 'bounding_boxes.txt'), 'r') as f:
            bbox_list = f.readlines()

        example_name_list = \
            ['bbox: ' + bbox.split('\n')[0].split(' ', maxsplit=1)[-1] +
             ' imagefile: ' + os.path.join(image_folder, imgf.split('\n')[0].split(' ')[-1])
             for bbox, imgf in zip(bbox_list, image_list)]


        train_list = ['{} '.format(
            np.random.choice(
                [int(item.split(os.path.sep)[-2].split('.')[0]),
                 np.random.randint(1, num_train_classes + 1)],
                p=[1. - self.swap_probability, self.swap_probability])) + item
            for item in example_name_list[:train_end_id]]

        eval_list = ['{} '.format(int(item.split(os.path.sep)[-2].split('.')[0])) + item
                     for item in example_name_list[train_end_id:]]


        example_names = {}

        example_names['train'] = train_list

        example_names['val'] = None

        example_names['eval'] = eval_list

        num_classes = {'train': 100, 'val': 0, 'eval': 100}

        shard_size = 1500

        dataset_mean = [[[0.485, 0.456, 0.406]]]
        dataset_std = [[[0.229, 0.224, 0.225]]]

        return example_names, num_classes, shard_size, dataset_mean, dataset_std


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        '''[bbox, image_path] = example_name.strip().split('imagefile: ')
        label = int(image_path.split(os.path.sep)[-2].split('.')[0])
        [xmin, ymin, b_width, b_height] = bbox.split(' ')[1:-1]'''

        [label, rest] = example_name.strip().split(' bbox: ')
        label = int(label)

        [bbox, image_path] = rest.strip().split(' imagefile: ')

        [xmin, ymin, b_width, b_height] = bbox.split(' ')

        with PIL.Image.open(image_path, 'r') as img:
            width, height = img.size

        xmin = float(xmin)
        xmax = min(xmin + float(b_width), float(width))

        ymin = float(ymin)
        ymax = min(ymin + float(b_height), float(height))


        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width,
                     'ymins': ymin/height, 'ymaxs': ymax/height,
                     'xmins': xmin/width, 'xmaxs': xmax/width}]

        return arg_list

class CUB200_2011LowNoise(CUB200_2011Noisy):

    def __init__(self, **kwargs):
        self.swap_probability = 0.1
        super(CUB200_2011LowNoise, self).__init__(**kwargs)

class CUB200_2011MediumNoise(CUB200_2011Noisy):

    def __init__(self, **kwargs):
        self.swap_probability = 0.2
        super(CUB200_2011MediumNoise, self).__init__(**kwargs)

class CUB200_2011HighNoise(CUB200_2011Noisy):

    def __init__(self, **kwargs):
        self.swap_probability = 0.4
        super(CUB200_2011HighNoise, self).__init__(**kwargs)