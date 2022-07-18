import tensorflow as tf
import numpy as np
import collections
import six
import os
import sys
import shutil
from multiprocessing import Pool
import json

from ..utilities import dataset_utils
from .samplers import Sampler
from .samplers import Random

class Dataset(object):
    # ==============================================================================
    # datasets specific methods
    # ==============================================================================

    # raw datasets organization
    # ==============================================================================
    def _reOrganizeDataset(self, base_dir):
        ''' should return 4-tuple:
        example_names: dictionary object with keys 'train', 'val', 'eval' where each value is list like
        names of the examples. The elements of the lists are of user choice depending on the implementation of
        _exampleFeatures method.

        num_classes: dictionary object with keys 'train', 'val', 'eval' where each value is a scalar indicating
        the number of distinct classes for each set

        shard_size: scalar indicating how many examples are encoded into a single file

        dataset_mean: statistical mean of the datasets to be possibly used in preprocessing.
        This can be scalar, vector or matrix. Simply returning .5 works for most preprocessing methods.


        example_names = {'train': list like names of the examples,
                         'val': list like names of the examples,
                         'eval': list like names of the examples}
        num_classes = {'train': scalar,
                       'val': scalar,
                       'eval': scalar}'''

        raise NotImplementedError(
            '''_reOrganizeDataset() is to be implemented in Dataset sub classes and
            it should return 4-tuple:
        example_names: dictionary object with keys 'train', 'val', 'eval' where each value is list like
        names of the examples. The elements of the lists are of user choice depending on the implementation of
        _exampleFeatures method.
        
        num_classes: dictionary object with keys 'train', 'val', 'eval' where each value is a scalar indicating
        the number of distinct classes for each set
        
        shard_size: scalar indicating how many examples are encoded into a single file
        
        dataset_mean: statistical mean of the datasets to be possibly used in preprocessing.
        This can be scalar, vector or matrix. Simply returning .5 works for most preprocessing methods.
        
        
        example_names = {'train': list like names of the examples,
                         'val': list like names of the examples,
                         'eval': list like names of the examples}
        num_classes = {'train': scalar,
                       'val': scalar,
                       'eval': scalar}''')

    # raw data to features
    # ==============================================================================
    def _exampleFeatures(self, example_name):
        raise NotImplementedError(
            '''_exampleFeatures() is to be implemented in Dataset sub classes
            and should return single or a list of dictionary object(s) of the key-value pairs:
            {'image_path': val, 'label': list_of_vals, 'height': val, 'width': val}
            and optionally with the augmentation of bbox coordinates list as:
            {'ymins': list_of_vals, 'ymaxs': list_of_vals, 'xmins': list_of_vals, 'xmaxs': list_of_vals}''')

    # ==============================================================================
    # shared methods
    # ==============================================================================

    @property
    def name(self):
        return str(self.__class__).split('.')[-1].split("'")[0]

    def __init__(self, dataset_dir=None, verbose=0):

        if dataset_dir is None:
            dataset_dir = os.path.join(os.path.expanduser('~'), '.cache/image_datasets')


        # class datasets organization
        self._valid_sets = ('train', 'trainval', 'val', 'eval')
        base_dir = os.path.join(dataset_dir, self.name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self._encoded_dataset_dir = os.path.join(base_dir, 'encoded_dataset')

        # check whether datasets is ready (preprocessed and archived as text and tf records)
        summary = self._getSummary(verbose=verbose)

        DatasetSize = collections.namedtuple('_DatasetSize', self._valid_sets)
        NumClasses = collections.namedtuple('_NumClasses', self._valid_sets)

        if summary is None:
            example_names, num_classes, shard_size, dataset_mean, dataset_std = self._reOrganizeDataset(base_dir)
            with tf.device('CPU:0'):
                dataset_size = self._createEncodedDataset(example_names, shard_size) # returns size of each subset

            # update info
            self.size = DatasetSize(**dataset_size, trainval=-1)
            self.num_classes = NumClasses(**num_classes, trainval=-1)
            self.mean = dataset_mean
            self.std = dataset_std

            # save summary
            summary = {'num_classes': self.num_classes._asdict(),
                       'size': self.size._asdict(),
                       'mean': self.mean,
                       'std': self.std}
            with open(os.path.join(self._encoded_dataset_dir, '.summary'), 'w') as f:
                json.dump(summary, f)
        else:
            # at this point info is available
            self.size = DatasetSize(**summary['size'])
            self.num_classes = NumClasses(**summary['num_classes'])
            self.mean = summary['mean']
            self.std = summary['std']

        NumClasses = \
            collections.namedtuple('_NumClasses', self.num_classes._fields + ('train_split',))

        DatasetSize = \
            collections.namedtuple('_DatasetSize', self.size._fields + ('train_split',))

        self.num_classes = NumClasses(**self.num_classes._asdict(), train_split=[])
        self.size = DatasetSize(**self.size._asdict(), train_split=[])

        self.split_eval_at = 0

    # retrieves datasets summary if exists
    # ==============================================================================
    def _getSummary(self, verbose=0):
        # check whether datasets is ready (preprocessed and archived as tf records)
        summary = None

        if not os.path.exists(os.path.join(self._encoded_dataset_dir, '.summary')):
            print(self.warning('Encoded dataset for {} is not available. It is to be created!'.format(self.name)))
        else:
            with open(os.path.join(self._encoded_dataset_dir, '.summary')) as f:
                summary = json.load(f)
                print_text = 'Encoded dataset for {} is available'.format(self.name)
                text_end = ' with the following properties:' if verbose > 1 else '!'
                print_text += text_end
                print(self.information(print_text))
                if verbose > 1:
                    print('num_classes: {}'.format(summary['num_classes']))
                    print('size: {}'.format(summary['size']))

        return summary

    # raw data to tf record or text record encoder
    # ==============================================================================
    def _exampleCoder(self, example_name, format='textample'):
        feature_dict_list = self._exampleFeatures(example_name)
        if not isinstance(feature_dict_list, (tuple, list)):
            feature_dict_list = [feature_dict_list]

        if format == 'textample':
            encoder = self._toTextample
        elif format == 'tfrecord':
            encoder = self._toTFRecordExample
        else:
            encoder = self._toTextample
            print(self.warning('Using textample encoding due to unsupported example encoding!'))


        examples = [encoder(**args) for args in feature_dict_list]

        return examples

    # creates text record example from the provided features
    # ==============================================================================
    def _toTextample(self, image_path, label, height, width,
                           ymins=0., ymaxs=1., xmins=0., xmaxs=1.):
        ''' example is of the form
        '<label> <img_path> <height> <width> [<bbox:ymin> <bbox:xmin> <bbox:ymax> <bbox:xmax>, ... ]' '''

        if not isinstance(ymins, (tuple, list)):
            ymins = [ymins]
            ymins = [int(round(ymin * height)) for ymin in ymins]
        if not isinstance(xmins, (tuple, list)):
            xmins = [xmins]
            xmins = [int(round(xmin * width)) for xmin in xmins]
        if not isinstance(ymaxs, (tuple, list)):
            ymaxs = [ymaxs]
            ymaxs = [int(round(ymax * height)) for ymax in ymaxs]
        if not isinstance(xmaxs, (tuple, list)):
            xmaxs = [xmaxs]
            xmaxs = [int(round(xmax * width)) for xmax in xmaxs]

        textample = '%s %s %s %s'%(str(label), image_path, str(height), str(width))

        bbox_encodings = ['%s %s %s %s'%(str(ymins[k]), str(xmins[k]), str(ymaxs[k]), str(xmaxs[k]))
                          for k in range(len(ymins))]

        for bbox_encoding in bbox_encodings:
            textample = textample + ' ' + bbox_encoding

        return textample

    # creates tf record example from the provided features
    # ==============================================================================
    def _toTFRecordExample(self, image_path, label, height, width,
                           ymins=0., ymaxs=1., xmins=0., xmaxs=1.):

        with tf.io.gfile.GFile(image_path, mode='br') as f:
            image_encoded = f.read()

        image_format = image_path.split('.')[-1]

        if not isinstance(ymins, (tuple, list)):
            ymins = [ymins]
            ymins = [int(round(ymin * height)) for ymin in ymins]
        if not isinstance(xmins, (tuple, list)):
            xmins = [xmins]
            xmins = [int(round(xmin * width)) for xmin in xmins]
        if not isinstance(ymaxs, (tuple, list)):
            ymaxs = [ymaxs]
            ymaxs = [int(round(ymax * height)) for ymax in ymaxs]
        if not isinstance(xmaxs, (tuple, list)):
            xmaxs = [xmaxs]
            xmaxs = [int(round(xmax * width)) for xmax in xmaxs]


        feature_dict = {
            'image/encoded': self.bytesFeature(image_encoded),
            'image/format': self.bytesFeature(image_format.encode('utf8')),
            'image/class/labels': self.int64Feature(label),
            'image/height': self.int64Feature(height),
            'image/width': self.int64Feature(width),
            'image/object/bbox/ymins': self.int64Feature(ymins),
            'image/object/bbox/ymaxs': self.int64Feature(ymaxs),
            'image/object/bbox/xmins': self.int64Feature(xmins),
            'image/object/bbox/xmaxs': self.int64Feature(xmaxs)
        }

        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    # creates encoded datasets for 3 sets (train, val, test)
    # ==============================================================================
    def _createEncodedDataset(self, example_names, shard_size):

        # clear target directory if necessary
        if os.path.exists(self._encoded_dataset_dir):
            print(self.information('Cleaning the encoded datasets...'))

            shutil.rmtree(self._encoded_dataset_dir)
            print(self.information('Cleaning has been done!'))

        # create save directories
        os.makedirs(self._encoded_dataset_dir)
        [os.makedirs(os.path.join(self._encoded_dataset_dir, dset))
         for dset in self._valid_sets if not (dset == 'trainval')]

        # create encoded datasets for each subset
        dataset_size = {}
        for example_set in self._valid_sets:
            if example_set == 'trainval':
                continue

            example_list = example_names[example_set]
            save_dir = os.path.join(self._encoded_dataset_dir, example_set)
            encoding_fn = \
                self._encodeDataset2Textamples if example_set == 'train' else self._encodeDataset2TFRecords
            if example_list != None:
                dataset_size[example_set] = int(self._encodeDataset(example_list=example_list,
                                                                    encoded_data_save_dir=save_dir,
                                                                    shard_size=shard_size,
                                                                    encoding_fn=encoding_fn,
                                                                    num_processes=4))
            else:
                dataset_size[example_set] = 0

        return dataset_size

    # converting the raw datasets into tfrecord datasets format
    # ==============================================================================
    def _encodeDataset(self,
                       example_list,
                       encoded_data_save_dir,
                       shard_size,
                       encoding_fn,
                       num_processes=4):

        num_examples = len(example_list)

        num_shards = np.ceil(num_examples / shard_size).astype(int)

        if num_shards <= num_processes:
            num_processes = num_shards

        shard_id_range = np.linspace(0, num_shards, num_processes + 1).astype(int)

        num_shards_in_processes = shard_id_range[1:] - shard_id_range[:-1]
        num_examples_in_processes = num_shards_in_processes * shard_size

        example_list_ranges = [[0, num_examples_in_processes[0]]]
        for n in range(1, num_processes - 1):
            start_indx = example_list_ranges[n - 1][1]
            end_indx = start_indx + num_examples_in_processes[n]
            example_list_ranges.append([start_indx, end_indx])

        example_list_ranges.append([example_list_ranges[num_processes - 2][1], num_examples])

        # Launch a process for each batch
        print(self.information('Launching %d processes for spacings: %s' % (num_processes, example_list_ranges)))
        sys.stdout.flush()

        # prepare arguments for the processes
        args = []
        for i in range(num_processes):
            example_list_for_process = example_list[example_list_ranges[i][0]:example_list_ranges[i][1]]
            id_offset_for_naming = shard_id_range[i] + 1  # start from 1
            process_name = 'process_%d' % (i + 1)
            args.append((encoded_data_save_dir,
                         example_list_for_process,
                         id_offset_for_naming,
                         shard_size,
                         num_shards,
                         process_name))


        with Pool(num_processes) as p:
            examples_processed = p.starmap(encoding_fn, args)

        total_examples_processed = np.sum(examples_processed)


        print(self.information('Finished processing %d images pairs in data set.' % total_examples_processed))
        sys.stdout.flush()

        return total_examples_processed

        # ==============================================================================

    # construction of textample datasets from the list of images in a single process
    # ==============================================================================
    def _encodeDataset2Textamples(self,
                                  encoded_data_save_dir,
                                  example_name_list,
                                  shard_id_offset,
                                  shard_size,
                                  total_shards,
                                  name):

        num_examples = len(example_name_list)

        num_shards = np.ceil(num_examples / shard_size).astype(int)

        shard_ranges = np.linspace(0, num_examples, num_shards + 1).astype(int)

        example_cntr = 0
        processed_example = 0
        for s in range(num_shards):
            shard_id = shard_id_offset + s
            output_filename = 'example_set_%05d-of-%05d.txt' % (shard_id, total_shards)
            output_file = os.path.join(encoded_data_save_dir, output_filename)

            # create the text file for the shard
            with open(output_file, 'wb') as textample_writer:
                for i in range(shard_ranges[s], shard_ranges[s + 1]):
                    if example_cntr % 100 == 0:
                        print(self.information('{}:processing example ({}) / ({})'.format(name, example_cntr + 1,
                                                                                      num_examples)))
                        sys.stdout.flush()

                    textamples = self._exampleCoder(example_name_list[i], format='textample')

                    if textamples != None:
                        for textample in textamples:
                            textample_writer.write((textample + os.linesep).encode('utf8'))
                            processed_example += 1
                    example_cntr += 1

        return processed_example

    # construction of tfrecord datasets from the list of images in a single process
    # ==============================================================================
    def _encodeDataset2TFRecords(self,
                                 encoded_data_save_dir,
                                 example_name_list,
                                 shard_id_offset,
                                 shard_size,
                                 total_shards,
                                 name):

        num_examples = len(example_name_list)

        num_shards = np.ceil(num_examples / shard_size).astype(int)

        shard_ranges = np.linspace(0, num_examples, num_shards + 1).astype(int)

        example_cntr = 0
        processed_example = 0
        for s in range(num_shards):
            shard_id = shard_id_offset + s
            output_filename = 'example_set_%05d-of-%05d.tfrecord' % (shard_id, total_shards)
            output_file = os.path.join(encoded_data_save_dir, output_filename)

            # create the tfrecord file for the shard
            with tf.io.TFRecordWriter(output_file) as tfrecord_writer:
                for i in range(shard_ranges[s], shard_ranges[s + 1]):
                    if example_cntr % 100 == 0:
                        print(self.information('{}:processing example ({}) / ({})'.format(name, example_cntr + 1, num_examples)))
                        sys.stdout.flush()

                    tf_examples = self._exampleCoder(example_name_list[i], format='tfrecord')

                    if tf_examples != None:
                        for tf_example in tf_examples:
                            tfrecord_writer.write(tf_example.SerializeToString())
                            processed_example += 1
                    example_cntr += 1

        return processed_example

    # ==============================================================================
    # TF-Feature conversion functions
    # ==============================================================================
    @staticmethod
    def int64Feature(values):
        """Returns a TF-Feature of int64s.
        Args:
            values: A scalar or list of values.
        Returns:
            a TF-Feature.
        """
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    @staticmethod
    def bytesFeature(values):
        """Returns a TF-Feature of bytes.
        Args:
            values: A string.
        Returns:
            a TF-Feature.
        """

        if not isinstance(values, (tuple, list)):
            values = [values]

        if isinstance(values, six.string_types):
            values = six.binary_type(values, encoding='utf-8')

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    @staticmethod
    def floatFeature(values):
        """Wrapper for inserting float features into Example proto."""
        if not isinstance(values, (tuple, list)):
            values = [values]

        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    # fancy printing
    # ===============
    @staticmethod
    def information(text):
        return '\033[3;34m' + 'INFO:Dataset: ' + text + '\033[0m'

    @staticmethod
    def warning(text):
        return '\033[35m' + 'WARNING:Dataset: ' + text + '\033[0m'

    @staticmethod
    def error(text):
        return '\033[1;31m' + 'EERROR:Dataset: ' + text + '\033[0m'

    # ==============================================================================
    # functions to parse tf example for classification
    # ==============================================================================

    # parsing example
    # ==============================================================================
    def _parseTFExample(self, example_proto):

        feature_dict = {'image/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                        'image/format': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
                        'image/height': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1),
                        'image/width': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1),
                        'image/class/labels': tf.io.VarLenFeature(dtype=tf.int64),
                        'image/object/bbox/ymins': tf.io.VarLenFeature(dtype=tf.int64),
                        'image/object/bbox/ymaxs': tf.io.VarLenFeature(dtype=tf.int64),
                        'image/object/bbox/xmins': tf.io.VarLenFeature(dtype=tf.int64),
                        'image/object/bbox/xmaxs': tf.io.VarLenFeature(dtype=tf.int64),
                        }

        parsed_features = tf.io.parse_single_example(example_proto, feature_dict)

        # image properties
        image_encoded = parsed_features['image/encoded']
        format = parsed_features['image/format']
        height = tf.cast(parsed_features['image/height'], tf.int32)
        width = tf.cast(parsed_features['image/width'], tf.int32)

        # bounding box arguments (get the first object info only in case exists more)
        ymin = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/ymins'])[0], tf.int32)
        ymax = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/ymaxs'])[0], tf.int32)
        xmin = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/xmins'])[0], tf.int32)
        xmax = tf.cast(tf.sparse.to_dense(parsed_features['image/object/bbox/xmaxs'])[0], tf.int32)

        # label
        label = tf.cast(tf.sparse.to_dense(parsed_features['image/class/labels'])[0], tf.int32)

        # image tensor
        if tf.strings.regex_full_match(tf.strings.upper(format), '(?:JPG|JPEG|PNG)'):
            if tf.equal(format, 'PNG'):
                image_tensor = tf.io.decode_png(image_encoded, channels=3)
            else:
                image_tensor = tf.io.decode_jpeg(image_encoded, channels=3)
        else:
            tf.print(self.warning('Unsupported image type, producing zeros image...'))
            image_tensor = tf.zeros(shape=(height, width, 3), dtype=tf.uint8)


        # convert to float
        image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)

        return {'image/data': image_tensor, 'image/label': label,
                'image/height': height, 'image/width': width,
                'image/object/bbox/ymin': ymin, 'image/object/bbox/ymax': ymax,
                'image/object/bbox/xmin': xmin, 'image/object/bbox/xmax': xmax}

    def _parseTextample(self, example_proto):

        ''' example is of the form
        '<label> <img_path> <height> <width> [<bbox:ymin> <bbox:xmin> <bbox:ymax> <bbox:xmax>, ... ]' '''

        args = tf.strings.split(example_proto, ' ')

        label = tf.strings.to_number(args[0], out_type=tf.int32)
        img_path = args[1]
        height = tf.strings.to_number(args[2], out_type=tf.int32)
        width = tf.strings.to_number(args[3], out_type=tf.int32)
        # bounding box arguments (get the first object info only in case exists more)
        ymin = tf.strings.to_number(args[4], out_type=tf.int32)
        xmin = tf.strings.to_number(args[5], out_type=tf.int32)
        ymax = tf.strings.to_number(args[6], out_type=tf.int32)
        xmax = tf.strings.to_number(args[7], out_type=tf.int32)

        # read and decode image
        image_data = tf.io.read_file(img_path)
        encoding = tf.strings.upper(tf.strings.split(img_path, '.')[-1])
        if tf.strings.regex_full_match(encoding, '(?:JPG|JPEG|PNG)'):
            if tf.equal(encoding, 'PNG'):
                image_tensor = tf.io.decode_png(image_data, channels=3)
            else:
                image_tensor = tf.io.decode_jpeg(image_data, channels=3)
        else:
            tf.print(self.warning('Unsupported image type, producing zeros image...'))
            image_tensor = tf.zeros(shape=(height, width, 3), dtype=tf.uint8)

        # convert to [0,1] range
        image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)

        return {'image/data': image_tensor, 'image/label': label,
                'image/height': height, 'image/width': width,
                'image/object/bbox/ymin': ymin, 'image/object/bbox/ymax': ymax,
                'image/object/bbox/xmin': xmin, 'image/object/bbox/xmax': xmax}

    # returns a parsing function which conditionally preprocesses the image
    # ==============================================================================
    def _prepareInput(self, example_proto, *args,
                      preprocess_fn=None, subset='train', crop_to_bbox=False):

        # parse examples
        if subset in ['train', 'trainval']:
            parsed_features = self._parseTextample(example_proto)
        else:
            parsed_features = self._parseTFExample(example_proto)

        # crop to bbox args
        if not crop_to_bbox:
            offset_height = tf.constant(0, dtype=tf.int32)
            offset_width = tf.constant(0, dtype=tf.int32)
            target_height = parsed_features['image/height']
            target_width = parsed_features['image/width']
        else:
            offset_height = parsed_features['image/object/bbox/ymin']
            target_height = parsed_features['image/object/bbox/ymax'] - offset_height

            offset_width = parsed_features['image/object/bbox/xmin']
            target_width = parsed_features['image/object/bbox/xmax'] - offset_width

        crop_bbox_kwargs = {'offset_height': offset_height, 'offset_width': offset_width,
                            'target_height': target_height, 'target_width': target_width}

        # get image data
        image_tensor = parsed_features['image/data']

        # possibly crop
        image_tensor = tf.image.crop_to_bounding_box(image_tensor, **crop_bbox_kwargs)

        # check whether auxiliary label exists
        if args != ():
            is_not_representative = tf.logical_not(tf.cast(args[-1], dtype=tf.bool))
        else:
            is_not_representative = tf.constant(True, dtype=tf.bool, shape=())

        preprocess_for_train = tf.logical_and(is_not_representative, tf.equal(subset, 'train'))

        image_tensor = preprocess_fn(image_tensor, training=preprocess_for_train)

        return image_tensor, parsed_features['image/label']

    def _datasetFromTFRecords(self, tf_record_dir):
        tf_record_list = sorted(os.listdir(tf_record_dir))
        data_sources = [os.path.join(tf_record_dir, tfrecord_file) for tfrecord_file in tf_record_list]

        # construct datasets and decode filenames and auxiliary features
        dataset = tf.data.TFRecordDataset(data_sources)

        return dataset

    def _datasetFromText(self, textample_dir, split_id=1, num_splits=1):
        file_list = os.listdir(textample_dir)
        data_sources = [os.path.join(textample_dir, file_name)
                        for file_name in file_list if file_name.endswith('.txt')]

        dataset = []
        for text_file in data_sources:
            with open(text_file, 'rb') as f:
                dataset += f.readlines()

        # split train and validation for k-fold (k = num_splits)
        class_label_ranges = np.linspace(0, self.num_classes.train, num_splits + 1).astype(int)
        val_beg = num_splits - split_id
        val_labels = [l for l in range(class_label_ranges[val_beg], class_label_ranges[val_beg + 1])]

        # remap labels
        train_dataset = []
        val_dataset = []
        num_excluded = len(val_labels)
        beg_lbl = val_labels[0]
        end_lbl = val_labels[-1]

        for d in dataset:
            lbl, item = d.strip().split(maxsplit=1)
            lbl = int(lbl)
            if lbl < beg_lbl:
                train_dataset.append(str(lbl).encode() + b' ' + item)
            elif lbl > end_lbl:
                train_dataset.append(str(lbl - num_excluded).encode() + b' ' + item)
            else:
                val_dataset.append(str(lbl - beg_lbl).encode() + b' ' + item)


        num_classes = self.num_classes.train - num_excluded
        if len(train_dataset) < 1:
            train_dataset = val_dataset # no split (i.e. 1-1 case)
            num_classes = self.num_classes.train

        if len(self.num_classes.train_split) < num_splits:
            diff = num_splits - len(self.num_classes.train_split)
            [self.num_classes.train_split.append(0) for _ in range(diff)]

        self.num_classes.train_split[split_id-1] = num_classes

        if len(self.size.train_split) < num_splits:
            diff = num_splits - len(self.size.train_split)
            [self.size.train_split.append(0) for _ in range(diff)]

        self.size.train_split[split_id-1] = len(train_dataset)

        return train_dataset, val_dataset

    def _toLabeledTuple(self, textample_dataset):
        # textample is of the form <label> <img_path> <height> <width> <bbox:ymin> <bbox:xmin> <bbox:ymax> <bbox:xmax>
        return [(textample.strip(), int(textample.split()[0])) for textample in textample_dataset]

    # input layer for something
    # ==============================================================================
    def _createInputPipeline(self,
                             subset='train',
                             split_id=1,
                             num_splits=1,
                             sampling_fn=None,
                             preprocess_fn=None,
                             crop_to_bbox=False,
                             batch_size=32):

        '''batch_size is used for evaluatuon pipleine and
        it is ignored for training if sampling_fn is provided '''

        # sampling from the datasets
        batch_size = batch_size
        if subset in ['train', 'trainval']:
            textample_dir = os.path.join(self._encoded_dataset_dir, 'train')
            train_dataset, val_dataset = self._datasetFromText(textample_dir,
                                                               split_id=split_id,
                                                               num_splits=num_splits)
            if subset == 'train':
                dataset = train_dataset
                if sampling_fn != None:
                    dataset = sampling_fn(dataset)
                    batch_size = sampling_fn.batch_size
                else:
                    dataset = tf.data.Dataset.from_tensor_slices([ds.strip() for ds in dataset])
                    dataset = dataset.repeat().batch(batch_size)
            else:
                dataset = tf.data.Dataset.from_tensor_slices([ds.strip() for ds in val_dataset]).batch(batch_size)
                self.split_eval_at = 0


        else: # for validation or test use tf records for better efficiency
            # get tf record filenames
            tf_record_dir = os.path.join(self._encoded_dataset_dir, subset)
            dataset = self._datasetFromTFRecords(tf_record_dir)
            dataset = dataset.batch(batch_size)

        # prepare (image, label) pairs
        prepare_fn = lambda *x: self._prepareInput(*x,
                                                   preprocess_fn=preprocess_fn,
                                                   subset=subset,
                                                   crop_to_bbox=crop_to_bbox)

        dataset_preprocessed = dataset.flat_map(
            lambda *x: tf.data.Dataset.from_tensor_slices(x).map(
                map_func=prepare_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size))


        return dataset_preprocessed


    # ==============================================================================
    # functions to be called by keras fit
    # ==============================================================================

    def makeBatch(self, subset,
                  split_id=1,
                  num_splits=1,
                  batch_size=32,
                  sampling_fn=None,
                  preprocess_fn=None,
                  crop_to_bbox=False):

        '''batch_size is used for evaluatuon pipline and
        it is ignored for training if sampling_fn is provided.'''

        if subset not in self._valid_sets:
            raise ValueError('example set must be in : {}'.format(self._valid_sets))

        if not isinstance(preprocess_fn, dataset_utils.ImagePreprocessing):
            raise ValueError('preprocess_fn should be of utils.ImagePreprocessing type.')

        if sampling_fn is None:
            sampling_fn = Random(batch_size=batch_size)
        else:
            if not isinstance(sampling_fn, Sampler):
                raise ValueError('sampling_fn should be of samplers.Sampler type.')

        batch_dataset = \
            self._createInputPipeline(subset=subset,
                                      split_id=split_id,
                                      num_splits=num_splits,
                                      sampling_fn=sampling_fn,
                                      preprocess_fn=preprocess_fn,
                                      crop_to_bbox=crop_to_bbox,
                                      batch_size=batch_size)


        return batch_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

