import tensorflow as tf

import os

from ...configs import default
from ...configs.config import CfgNode as CN

# EfficientNet
EfficientNet_cfg = CN()
EfficientNet_cfg.arch_parameters = CN()
EfficientNet_cfg.arch_parameters.use_pretrained = True
EfficientNet_cfg.arch_parameters.model_file = '../pretrained_models/EfficientNet.h5'
EfficientNet_cfg.arch_parameters.freeze_bn = True
EfficientNet_cfg.arch_parameters.end_layer = None    # use up to final fc layer
EfficientNet_cfg.input_parameters = CN()
EfficientNet_cfg.input_parameters.image_scaler = 255.0
EfficientNet_cfg.input_parameters.reverse_channels = False

default.cfg.model.backbone.EfficientNet = EfficientNet_cfg

def EfficientNet(model_filepath, end_layer=None, freeze_bn=True, input_shape=(None, None, None, 3), name=None):

    model_file = os.path.join(model_filepath, 'EfficientNet.h5')

    base_model = tf.keras.models.load_model(model_file, compile=False)

    if end_layer is not None:
        base_model = tf.keras.models.Model(inputs=base_model.input,
                                           outputs=base_model.get_layer(name=end_layer).output)

    # FREEZING
    if freeze_bn:
        for layer in base_model.layers:
            if 'BatchNormalization' in str(layer.__class__):
                layer.trainable = False

    #base_model.input.set_shape(shape=input_shape[1:])

    return base_model