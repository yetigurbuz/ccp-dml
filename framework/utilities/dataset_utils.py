import tensorflow as tf

from ..configs import default
from ..configs.config import CfgNode as CN

def resizeImageKeepAspectRatio(image_tensor, target_size):

    lo_dim = tf.cast(tf.reduce_max(target_size), tf.float32)

    # Take width/height
    initial_width = tf.shape(image_tensor)[0]
    initial_height = tf.shape(image_tensor)[1]

    # Take the greater value, and use it for the ratio
    min_ = tf.minimum(initial_width, initial_height)
    ratio = tf.cast(min_, tf.float32) / lo_dim #tf.constant(lo_dim, dtype=tf.float32)

    new_width = tf.cast(tf.cast(initial_width, tf.float32) / ratio, tf.int32)
    new_height = tf.cast(tf.cast(initial_height, tf.float32) / ratio, tf.int32)

    return tf.image.resize(image_tensor, [new_width, new_height], method=tf.image.ResizeMethod.BILINEAR)

class ImagePreprocessing(object):

    def __init__(self, **kwargs):
        pass

    @tf.function
    def __call__(self, image, training=tf.constant(False, dtype=tf.bool), **kwargs):

        image = tf.cond(training,
                        true_fn=lambda: self.trainPreprocess(image, **kwargs),
                        false_fn=lambda: self.testPreprocess(image, **kwargs))

        return image

    def trainPreprocess(self, image, **kwargs):
        raise NotImplementedError(
            'trainPreprocess() is to be implemented in ImagePreprocessing sub classes')

    def testPreprocess(self, image, **kwargs):
        raise NotImplementedError(
            'testPreprocess() is to be implemented in ImagePreprocessing sub classes')

# MNISTPreprocessing
MNISTPreprocessing_cfg = CN()
MNISTPreprocessing_cfg.initial_resize = (40, 40)
MNISTPreprocessing_cfg.out_image_size = (28, 28)
MNISTPreprocessing_cfg.out_image_channels = 3
default.cfg.dataset.preprocessing.MNISTPreprocessing = MNISTPreprocessing_cfg

class MNISTPreprocessing(ImagePreprocessing):
    def __init__(self, mean_image=.5, initial_resize=(40, 40), out_image_size=(28, 28), **kwargs):
        super(MNISTPreprocessing, self).__init__(**kwargs)
        self._mean_image = tf.reshape(tf.constant(mean_image, dtype=tf.float32), shape=(28, 28, 1))
        self._initial_resize = initial_resize
        self._out_image_size = out_image_size
        self.out_image_size = list(self._out_image_size) + [1]

    def trainPreprocess(self, image, **kwargs):
        """Preprocess a single image in [height, width, depth] layout."""
        # Pad 4 pixels on each dimension of feature map, done in mini-batch

        image_tensor = image - self._mean_image
        #image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, self._initial_resize[0], self._initial_resize[1])
        #image_tensor = tf.image.random_crop(image_tensor, list(self._out_image_size))
        #image_tensor = tf.image.random_flip_left_right(image_tensor)

        return tf.reduce_mean(image_tensor, axis=-1, keepdims=True)

    def testPreprocess(self, image, **kwargs):
        """pass image as is"""

        image_tensor = image - self._mean_image
        #image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, 40, 40)
        #image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, *self._out_image_size)

        return tf.reduce_mean(image_tensor, axis=-1, keepdims=True)

# CifarPreprocessing
CifarPreprocessing_cfg = CN()
CifarPreprocessing_cfg.initial_resize = (40, 40)
CifarPreprocessing_cfg.out_image_size = (32, 32)
CifarPreprocessing_cfg.out_image_channels = 3
CifarPreprocessing_cfg.recenter = True
CifarPreprocessing_cfg.std_normalize = False

default.cfg.dataset.preprocessing.CifarPreprocessing = CifarPreprocessing_cfg

class CifarPreprocessing(ImagePreprocessing):
    def __init__(self,
                 mean_image=.5,
                 std_image=1.,
                 recenter=True,
                 std_normalize=False,
                 initial_resize=(40, 40),
                 out_image_size=(32, 32),
                 image_scaler=1.,
                 **kwargs
                 ):
        super(CifarPreprocessing, self).__init__(**kwargs)

        if mean_image is None:
            mean_image = 0.5
        if (std_image is None) or (not std_normalize):
            std_image = 1.
        if not recenter:
            mean_image = 0.0

        self._mean_image = tf.constant(mean_image, dtype=tf.float32)
        self._std_image = tf.constant(std_image, dtype=tf.float32)
        self._initial_resize = initial_resize
        self._out_image_size = out_image_size
        self.out_image_size = list(self._out_image_size) + [3]

        self._image_scaler = image_scaler

        self._random_crop = True
        if self._out_image_size == self._initial_resize:
            self._random_crop = False



    def trainPreprocess(self, image, **kwargs):
        """Preprocess a single image in [height, width, depth] layout."""

        # normalize
        image_tensor = self._image_scaler * (image - self._mean_image) / self._std_image

        if self._random_crop:
            # Pad pixels on each dimension of feature map, done in mini-batch
            image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, *self._initial_resize)
            image_tensor = tf.image.random_crop(image_tensor, self.out_image_size)

        image_tensor = tf.image.random_flip_left_right(image_tensor)

        return image_tensor

    def testPreprocess(self, image, **kwargs):
        """pass image as is"""

        # normalize
        image_tensor = self._image_scaler * (image - self._mean_image) / self._std_image

        #image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, *self._initial_resize)
        image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, *self._out_image_size)

        return image_tensor

# DMLPreprocessing
DMLPreprocessing_cfg = CN()
DMLPreprocessing_cfg.short_axis_initial_resize = 256
DMLPreprocessing_cfg.out_image_size = (227, 227)
DMLPreprocessing_cfg.out_image_channels = 3
DMLPreprocessing_cfg.scale_range = (.16, 1.)
DMLPreprocessing_cfg.aspect_ratio_range = (.75, 1.33)
DMLPreprocessing_cfg.recenter = True
DMLPreprocessing_cfg.std_normalize = False

default.cfg.dataset.preprocessing.DMLPreprocessing = DMLPreprocessing_cfg

class DMLPreprocessing(ImagePreprocessing):
    def __init__(self,
                 mean_image=[[[0.485, 0.456, 0.406]]], # imagenet mean
                 std_image=[[[0.229, 0.224, 0.225]]], # imagenet std
                 recenter=True,
                 std_normalize=False,
                 short_axis_initial_resize=256,
                 scale_range=(.16, 1.),
                 aspect_ratio_range=(.75, 1.33),
                 out_image_size=(227, 227),
                 image_scaler=2.,
                 reverse_channels=False,
                 **kwargs):

        super(DMLPreprocessing, self).__init__(**kwargs)
        if mean_image is None:
            mean_image = 0.5
        if (std_image is None) or (not std_normalize):
            std_image = 1.
        if not recenter:
            mean_image = 0.5


        self._mean_image = tf.constant(mean_image, dtype=tf.float32)
        self._std_image = tf.constant(std_image, dtype=tf.float32)
        self._short_axis_initial_resize = short_axis_initial_resize
        self._scale_range = scale_range
        self._aspect_ratio_range = aspect_ratio_range
        self._out_image_size = out_image_size
        self.out_image_size = list(self._out_image_size) + [3]
        self._value_scaler = image_scaler
        self._reverse_channels = reverse_channels

    def trainPreprocess(self, image, **kwargs):
        """Preprocess a single image in [height, width, depth] layout."""

        #image_tensor = resizeImageKeepAspectRatio(image, target_size=self._short_axis_initial_resize)
        image_tensor = image
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])

        bbox_begin, bbox_size, _ = (
            tf.image.sample_distorted_bounding_box(tf.shape(image_tensor),
                                                   bounding_boxes=bbox,
                                                   min_object_covered=0,
                                                   aspect_ratio_range=self._aspect_ratio_range,
                                                   area_range=self._scale_range,
                                                   max_attempts=100,
                                                   use_image_if_no_bounding_boxes=True))

        # crop the image to the specified bounding box.
        cropped_image = tf.slice(image_tensor, bbox_begin, bbox_size)

        # resize to output size
        resized_image = tf.image.resize(cropped_image, self._out_image_size, method=tf.image.ResizeMethod.BILINEAR)

        # image_tensor = resizeImageKeepAspectRatio(image_tensor, target_size=self._out_image_size)
        # image_tensor = tf.image.random_crop(image_tensor, list(self._out_image_size) + [3])

        flipped_image = tf.image.random_flip_left_right(resized_image)
        image_tensor = self._value_scaler * (flipped_image - self._mean_image) / self._std_image

        if self._reverse_channels:
            image_tensor = tf.reverse(image_tensor, axis=[2])

        return image_tensor

    def testPreprocess(self, image, **kwargs):
        """pass image as is"""
        image_tensor = resizeImageKeepAspectRatio(image, target_size=self._short_axis_initial_resize)
        image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, *self._out_image_size)
        image_tensor = self._value_scaler * (image_tensor - self._mean_image) / self._std_image

        if self._reverse_channels:
            image_tensor = tf.reverse(image_tensor, axis=[2])

        return image_tensor

class DMLPrep(ImagePreprocessing):
    def __init__(self,
                 mean_image=[[[0.485, 0.456, 0.406]]], # imagenet mean
                 std_image=[[[0.229, 0.224, 0.225]]], # imagenet std
                 short_axis_initial_resize=256,
                 scale_range=(.16, 1.),
                 aspect_ratio_range=(.75, 1.33),
                 out_image_size=(227, 227),
                 image_scaler=2.,
                 reverse_channels=False,
                 **kwargs):

        super(DMLPrep, self).__init__(**kwargs)
        if mean_image is None:
            mean_image = 0.5
        if std_image is None:
            std_image = 1.
        self._mean_image = tf.constant(mean_image, dtype=tf.float32)
        self._std_image = tf.constant(std_image, dtype=tf.float32)
        self._short_axis_initial_resize = short_axis_initial_resize
        self._scale_range = scale_range
        self._aspect_ratio_range = aspect_ratio_range
        self._out_image_size = out_image_size
        self._value_scaler = image_scaler
        self._reverse_channels = reverse_channels

    def trainPreprocess(self, image, **kwargs):
        """Preprocess a single image in [height, width, depth] layout."""

        return self.testPreprocess(image, **kwargs)

    def testPreprocess(self, image, **kwargs):
        """pass image as is"""
        image_tensor = resizeImageKeepAspectRatio(image, target_size=self._short_axis_initial_resize)
        image_tensor = tf.image.resize_with_crop_or_pad(image_tensor, *self._out_image_size)
        image_tensor = self._value_scaler * (image_tensor - self._mean_image) / self._std_image

        if self._reverse_channels:
            image_tensor = tf.reverse(image_tensor, axis=[2])

        return image_tensor
