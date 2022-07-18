import tensorflow as tf

import os

from ...configs import default
from ...configs.config import CfgNode as CN

# BN-Inception (inception v2)
BNInception_cfg = CN()
BNInception_cfg.arch_parameters = CN()
BNInception_cfg.arch_parameters.use_pretrained = True
BNInception_cfg.arch_parameters.model_file = 'https://drive.google.com/u/0/uc?id=1WyYw5-WKwjWWyfXizoqmYIA2AEPB_vY_&export=download'
BNInception_cfg.arch_parameters.pretrained_model_file = ''
BNInception_cfg.arch_parameters.pretrained_name = 'BN-Inception'
BNInception_cfg.arch_parameters.freeze_bn = True
BNInception_cfg.arch_parameters.end_layer = None    # use up to final fc layer
BNInception_cfg.input_parameters = CN()
BNInception_cfg.input_parameters.image_scaler = 255.0
BNInception_cfg.input_parameters.reverse_channels = True

default.cfg.model.backbone.BNInception = BNInception_cfg

def BNInception(use_pretrained=False,
                model_file='',
                end_layer=None,
                freeze_bn=True,
                pretrained_model_file='',
                pretrained_name='BN-Inception',
                input_shape=(None, None, None, 3),
                name='BNInception'):


    # get architecture with image-net pretrained weights

    download_url = model_file

    os.makedirs('../pretrained_models', exist_ok=True)

    path_to_file = tf.keras.utils.get_file(
        fname='../pretrained_models/BNInception.h5',
        origin=download_url,
        cache_subdir='../pretrained_models',
        cache_dir='../pretrained_models')


    base_model = tf.keras.models.load_model(path_to_file, compile=False)

    if end_layer is not None:
        base_model = tf.keras.models.Model(inputs=base_model.input,
                                           outputs=base_model.get_layer(name=end_layer).output)

    if not use_pretrained:
        base_model = tf.keras.models.clone_model(base_model)
        freeze_bn = False
    elif pretrained_model_file.endswith('.h5'):
        pretrained_model = tf.keras.models.load_model(pretrained_model_file, compile=False)
        inception_backbone = pretrained_model.get_layer(pretrained_name)
        base_model.set_weights(inception_backbone.get_weights())

    # FREEZING
    if freeze_bn:
        for layer in base_model.layers:
            if 'BatchNormalization' in str(layer.__class__):
                layer.trainable = False

    #base_model.input.set_shape(shape=input_shape[1:])

    return base_model

def BNInceptiON(model_filepath, end_layer=None, freeze_bn=True, **kwargs):

    model_file = os.path.join(model_filepath, 'BNInception.h5')

    base_model = tf.keras.models.load_model(model_file, compile=False)


    if end_layer is not None:
        base_model = tf.keras.models.Model(inputs=base_model.input,
                                           outputs=base_model.get_layer(name=end_layer).output)

    if not kwargs['use_pretrained']:
        base_model = tf.keras.models.clone_model(base_model)
        freeze_bn = False
        print('\nMODEL: Training from scratch!\n')

    # FREEZING
    if freeze_bn:
        for layer in base_model.layers:
            if 'BatchNormalization' in str(layer.__class__):
                layer.trainable = False


    model = vip_model(base_model=base_model, **kwargs)


    return model
