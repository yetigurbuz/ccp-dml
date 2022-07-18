import numpy as np
import tensorflow as tf
from framework.datasets.samplers import MPerClass



class TokenCorpus(object):
    def __init__(self, num_classes, feature_per_class, num_shared_feature=3):

        self.shared = tuple([str(k) for k in range(num_shared_feature)])

        k = num_shared_feature
        self.classes = tuple([tuple([str(k + l + m * feature_per_class)
                                     for l in range(feature_per_class)])
                              for m in range(num_classes)])

        self.num_classes = num_classes

        self.vocabulary_size = num_classes * feature_per_class + num_shared_feature




    def createDataset(self, bag_size=10, beta=0.7, sample_per_class=8, classes_per_batch=4, batches_per_epoch=512):

        @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
        def parsing_fn(example_proto):
            ''' example is of the form
                    '<label> <feat_0> <feat_1> ... <feat_n>' '''


            args = tf.strings.split(example_proto, ' ')
            label = tf.strings.to_number(args[0], out_type=tf.int32)
            tokens = tf.strings.to_number(args[1:], out_type=tf.int32)

            one_hot_tokens = tf.one_hot(tokens, depth=self.vocabulary_size, dtype=tf.float32)

            return one_hot_tokens, label


        background_pool = np.array(self.shared)
        class_pool = np.array(self.classes)

        num_classes = len(class_pool)

        train_dataset_str = []
        val_dataset_str = []
        train_sample_size = batches_per_epoch * sample_per_class
        val_sample_size = 128
        sample_size = train_sample_size + val_sample_size
        l = 0
        for pool in class_pool:

            # training
            class_feats = pool[np.random.randint(low=0, high=len(pool), size=(sample_size, bag_size))]
            bg_feats = background_pool[np.random.randint(low=0, high=len(background_pool), size=(sample_size, bag_size))]

            num_background_samples = np.maximum(np.round((0.1 * np.random.randn(sample_size) + beta) * bag_size), 1)
            num_background_samples = np.minimum(num_background_samples, bag_size - 1).astype(np.int)

            samples = [str(int(l)) + ' ' +
                       ' '.join(b[:k].tolist()) + ' ' +
                       ' '.join(c[:(bag_size - k)])
                       for b, c, k in zip(bg_feats, class_feats, num_background_samples)]


            train_dataset_str += samples[:train_sample_size]
            val_dataset_str += samples[-val_sample_size:]

            l += 1

        sampler = MPerClass(classes_per_batch=classes_per_batch, sample_per_class=sample_per_class)

        train_dataset = sampler(train_dataset_str).flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        train_dataset = train_dataset.map(parsing_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(sampler.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset_str)
        val_dataset = val_dataset.map(parsing_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(val_sample_size)

        return train_dataset, val_dataset


