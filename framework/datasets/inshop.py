import os
import tensorflow as tf
from .dataset_base import Dataset


class InShop(Dataset):

    def __init__(self, **kwargs):
        super(InShop, self).__init__(**kwargs)
        self.split_eval_at = 14218

    def _reOrganizeDataset(self, base_dir):

        base_dir = os.path.join(os.path.abspath(os.getcwd()), base_dir)

        main_folder = os.path.join(base_dir, 'InShopClothesRetrievalBenchmark')
        if not os.path.exists(main_folder):
            download_url = \
                'https://download.wetransfer.com/eugv/d4725cdb9f9ef41b8cbe2c4702b29db820220514234412/9cdf9f4fbd30f545858a7013914003967c133498/InShopClothesRetrievalBenchmark.zip?token=eyJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2NTI1NzE5MzcsImV4cCI6MTY1MjU3MjUzNywidW5pcXVlIjoiZDQ3MjVjZGI5ZjllZjQxYjhjYmUyYzQ3MDJiMjlkYjgyMDIyMDUxNDIzNDQxMiIsImZpbGVuYW1lIjoiSW5TaG9wQ2xvdGhlc1JldHJpZXZhbEJlbmNobWFyay56aXAiLCJ3YXliaWxsX3VybCI6Imh0dHA6Ly9zdG9ybS1pbnRlcm5hbC5zZXJ2aWNlLmV1LXdlc3QtMS53ZXRyYW5zZmVyLm5ldC9hcGkvd2F5YmlsbHM_c2lnbmVkX3dheWJpbGxfaWQ9ZXlKZmNtRnBiSE1pT25zaWJXVnpjMkZuWlNJNklrSkJhSE5MZDJoYU1HYzFSVUZSUVQwaUxDSmxlSEFpT2lJeU1ESXlMVEExTFRFMVZEQXdPalUxT2pNM0xqQXdNRm9pTENKd2RYSWlPaUozWVhsaWFXeHNYMmxrSW4xOS0tZjFkZjZmMTAxZTIzMWM3ZjY4NGE1NzFkNzg0M2U3ZGU2ZmY1NjEyZmU1MTAwMjdiMjJjNjg2NDRjMjQyYzQ4OSIsImZpbmdlcnByaW50IjoiOWNkZjlmNGZiZDMwZjU0NTg1OGE3MDEzOTE0MDAzOTY3YzEzMzQ5OCIsImNhbGxiYWNrIjoie1wiZm9ybWRhdGFcIjp7XCJhY3Rpb25cIjpcImh0dHA6Ly9mcm9udGVuZC5zZXJ2aWNlLmV1LXdlc3QtMS53ZXRyYW5zZmVyLm5ldC93ZWJob29rcy9iYWNrZW5kXCJ9LFwiZm9ybVwiOntcInRyYW5zZmVyX2lkXCI6XCJkNDcyNWNkYjlmOWVmNDFiOGNiZTJjNDcwMmIyOWRiODIwMjIwNTE0MjM0NDEyXCIsXCJkb3dubG9hZF9pZFwiOjE1MjM2Nzk1NTg3LFwicmVjaXBpZW50X2lkXCI6XCJmNjk3ZTVlMjQ2NzA5MWFmMTkyNmZmODg4ZjdkZmY2YjIwMjIwNTE0MjM0NDMwXCJ9fSJ9.r3r24l2JfuXx5S34pzicKxAgqAEJOrrUX69tgo-maog&cf=y'

            fname = main_folder

            tf.keras.utils.get_file(fname=os.path.join(base_dir, fname),
                                    origin=download_url,
                                    cache_subdir=base_dir,
                                    extract=True)

        # original datasets organization
        img_folder = os.path.join(base_dir, 'InShopClothesRetrievalBenchmark/Img')
        splits = os.path.join(base_dir, 'InShopClothesRetrievalBenchmark/Eval/list_eval_partition.txt')
        bbox_annotations = os.path.join(base_dir, 'InShopClothesRetrievalBenchmark/Anno/list_bbox_inshop.txt')
        re_organized = os.path.join(base_dir, 'InShopClothesRetrievalBenchmark/Eval/re_organized')

        if not os.path.exists('{}_train.txt'.format(re_organized)):

            with open(splits, 'r') as f:
                image_list = f.readlines()[2:]

            with open(bbox_annotations, 'r') as f:
                bbox_list = f.readlines()[2:]
            bbox_string = '-'.join(bbox_list)

            label_translator = {}
            new_label = 0

            train_list = []
            query_list = []
            gallery_list = []
            for i, item in enumerate(image_list):

                if i%(len(image_list) // 100) == 0:
                    print(self.information('re-organizing: {} % ...'.format(round(i/len(image_list) * 100))))

                img_path, rest = item.split(' ', maxsplit=1)
                label = int(rest.split('id_')[-1].split(' ')[0])
                # xmin ymin xmax ymax
                bbox = bbox_string.split(img_path)[-1].split('-')[0].strip().split(' ')[2:]
                if 'train' in item:
                    if not (label in label_translator):
                        label_translator[label] = new_label
                        new_label += 1
                    item_label = label_translator[label]
                    train_list.append('{} bbox: '.format(item_label) + ' '.join(bbox)
                                      + ' imagefile: ' + os.path.join(img_folder, img_path))
                elif 'query' in item:
                    query_list.append('{} bbox: '.format(label) + ' '.join(bbox)
                                      + ' imagefile: ' + os.path.join(img_folder, img_path))
                else:
                    gallery_list.append('{} bbox: '.format(label) + ' '.join(bbox)
                                      + ' imagefile: ' + os.path.join(img_folder, img_path))

            with open('{}_train.txt'.format(re_organized), 'w+') as f:
                for item in train_list:
                    f.write("%s\n" % item)
            with open('{}_query.txt'.format(re_organized), 'w+') as f:
                for item in query_list:
                    f.write("%s\n" % item)
            with open('{}_gallery.txt'.format(re_organized), 'w+') as f:
                for item in gallery_list:
                    f.write("%s\n" % item)
        else:
            with open('{}_train.txt'.format(re_organized), 'r') as f:
                train_list = f.readlines()
            with open('{}_query.txt'.format(re_organized), 'r') as f:
                query_list = f.readlines()
            with open('{}_gallery.txt'.format(re_organized), 'r') as f:
                gallery_list = f.readlines()


        example_names = {}

        example_names['train'] = train_list

        example_names['val'] = None

        example_names['eval'] = query_list + gallery_list

        num_classes = {'train': 3997, 'val': 0, 'eval': 3985}

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


        width = 256
        height = 256

        xmin = float(xmin)
        xmax = float(xmax)

        ymin = float(ymin)
        ymax = float(ymax)


        arg_list = [{'image_path': image_path, 'label': label, 'height': height, 'width': width,
                     'ymins': ymin/height, 'ymaxs': ymax/height,
                     'xmins': xmin/width, 'xmaxs': xmax/width}]

        return arg_list




