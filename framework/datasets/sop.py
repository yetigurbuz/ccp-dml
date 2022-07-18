import tensorflow as tf
import PIL
import os

from .dataset_base import Dataset

class SOP(Dataset):

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        # original datasets organization
        main_folder = os.path.join(base_dir, 'Stanford_Online_Products')

        # download and extract datasets if not exists
        if not os.path.exists(main_folder):
            download_url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
            fname = os.path.split(download_url)[-1]

            print(self.information('Downloading and extracting the datasets...'))
            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        example_names = {}

        def line2Name(l):
            s = l.strip().split()
            return s[1] + ' ' + os.path.join(main_folder, s[-1])

        with open(os.path.join(main_folder, 'Ebay_train.txt'), 'r') as f:
            image_list = f.readlines()[1:]

        example_names['train'] = [line2Name(l) for l in image_list]

        example_names['val'] = None

        with open(os.path.join(main_folder, 'Ebay_test.txt'), 'r') as f:
            image_list = f.readlines()[1:]

        example_names['eval'] = [line2Name(l) for l in image_list]

        num_classes = {'train': 11318, 'val': 0, 'eval': 11316}

        shard_size = 15000

        dataset_mean = [[[0.485, 0.456, 0.406]]]
        dataset_std = [[[0.229, 0.224, 0.225]]]

        return example_names, num_classes, shard_size, dataset_mean, dataset_std


    def _exampleFeatures(self, example_name):
        # returns a list of arguments for tf examples extracted from the data of example_name

        [label, image_path] = example_name.split(' ')
        label = int(label)

        with PIL.Image.open(image_path, 'r') as img:
            width, height = img.size


        arg_list = [{'image_path': image_path, 'label': label - 1, 'height': height, 'width': width}]

        return arg_list



def createSOPtxt(main_folder_path):

    def line2Name(l):
        s = l.strip().split()
        return os.path.join(main_folder_path, s[-1]) + ' ' + str(int(s[1]) - 1)

    with open(os.path.join(main_folder_path, 'Ebay_train.txt'), 'r') as f:
        image_list = f.readlines()[1:]

    example_names_train = [line2Name(l) for l in image_list]

    with open(os.path.join(main_folder_path, 'Ebay_test.txt'), 'r') as f:
        image_list = f.readlines()[1:]

    example_names_test = [line2Name(l) for l in image_list]

    base_name = os.path.join(main_folder_path, 'sop')
    with open('{}_train.txt'.format(base_name), 'w+') as f:
        for item in example_names_train:
            f.write("%s\n" % item)
    with open('{}_test.txt'.format(base_name), 'w+') as f:
        for item in example_names_test:
            f.write("%s\n" % item)
