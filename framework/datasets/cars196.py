import tensorflow as tf
from scipy import io as scio
import PIL
import os

from .dataset_base import Dataset

class Cars196(Dataset):

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original datasets organization
        image_folder = os.path.join(base_dir, 'car_ims')
        annotations_file = os.path.join(base_dir, 'cars_annos.mat')

        print(self.information('Downloading and extracting the datasets...'))
        # download and extract datasets if not exists
        if not os.path.exists(image_folder):
            download_url = 'http://ai.stanford.edu/~jkrause/car196/car_ims.tgz'
            fname = os.path.split(download_url)[-1]

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        if not os.path.exists(annotations_file):
            download_url = 'http://ai.stanford.edu/~jkrause/car196/cars_annos.mat'
            fname = os.path.split(download_url)[-1]

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url)

        annotations = scio.loadmat(annotations_file)['annotations'].ravel()

        def item2example_name(item):
            img_path = os.path.join(base_dir, item[0][0])
            label = str(item[-2][0][0])
            bbox = str(item[1][0][0]) + ' ' + str(item[2][0][0]) + ' ' + str(item[3][0][0]) + ' ' + str(item[4][0][0]) + ' '
            return label + ' bbox: ' + bbox + 'imagefile: ' + img_path

        # specify the splits (labels 1~98 for train, 99~196 for test)
        example_name_lists = [[], []]
        [example_name_lists[item[-2][0][0] > 98].append(item2example_name(item)) for item in annotations]

        example_names = {}

        example_names['train'] = example_name_lists[0]

        example_names['val'] = None

        example_names['eval'] = example_name_lists[1]

        num_classes = {'train': 98, 'val': 0, 'eval': 98}

        shard_size = 1500

        dataset_mean = [[[0.485, 0.456, 0.406]]]
        dataset_std = [[[0.229, 0.224, 0.225]]]

        return example_names, num_classes, shard_size, dataset_mean, dataset_std


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        [label, rest] = example_name.strip().split(' bbox: ')
        label = int(label)

        [bbox, image_path] = rest.strip().split(' imagefile: ')

        [xmin, ymin, xmax, ymax] = bbox.split(' ')


        with PIL.Image.open(image_path, 'r') as img:
            width, height = img.size

        xmin = float(xmin)
        xmax = float(xmax)

        ymin = float(ymin)
        ymax = float(ymax)


        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width,
                     'ymins': ymin/height, 'ymaxs': ymax/height,
                     'xmins': xmin/width, 'xmaxs': xmax/width}]

        return arg_list




