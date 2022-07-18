import tensorflow as tf
from ...layers import resMap, resBlock
import os

from ...configs import default
from ...configs.config import CfgNode as CN

# ResNet50V2
ResNet50V2_cfg = CN()
ResNet50V2_cfg.arch_parameters = CN()
ResNet50V2_cfg.arch_parameters.use_pretrained = True
ResNet50V2_cfg.arch_parameters.pretrained_model_file = ''
ResNet50V2_cfg.arch_parameters.freeze_bn = True
ResNet50V2_cfg.arch_parameters.end_layer = None    # use up to final fc layer
ResNet50V2_cfg.input_parameters = CN()
ResNet50V2_cfg.input_parameters.image_scaler = 255.0
ResNet50V2_cfg.input_parameters.reverse_channels = True



default.cfg.model.backbone.ResNet50V2 = ResNet50V2_cfg


def ResNet50V2(pretrained_model_file='', use_pretrained=True, end_layer=None, freeze_bn=True, input_shape=(None, None, 3), name=None):


    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights='imagenet',
                                                         input_shape=input_shape[-3:])

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




# CifarResNet
CifarResNet_cfg = CN()
CifarResNet_cfg.arch_parameters = CN()
CifarResNet_cfg.arch_parameters.use_pretrained = False
CifarResNet_cfg.arch_parameters.pretrained_model_file = ''
CifarResNet_cfg.arch_parameters.pretrained_name = 'CifarResNet'    # use up to final fc layer
CifarResNet_cfg.arch_parameters.stages = 3
CifarResNet_cfg.arch_parameters.blocks = 2
CifarResNet_cfg.arch_parameters.block_repeats = 1
CifarResNet_cfg.arch_parameters.final_activation = True
CifarResNet_cfg.arch_parameters.classification_head = False
CifarResNet_cfg.arch_parameters.num_classes = None
CifarResNet_cfg.arch_parameters.block_parameters = CN()
CifarResNet_cfg.arch_parameters.block_parameters.spatial_aggregation = 'skip'
CifarResNet_cfg.arch_parameters.block_parameters.pre_activation = True
CifarResNet_cfg.arch_parameters.block_parameters.pre_activation_block = 'BN_ReLU'
CifarResNet_cfg.arch_parameters.block_parameters.mid_activation = True
CifarResNet_cfg.arch_parameters.block_parameters.mid_activation_block = 'BN_ReLU'
CifarResNet_cfg.arch_parameters.block_parameters.post_activation = True
CifarResNet_cfg.arch_parameters.block_parameters.post_activation_block = 'BN_ReLU'
CifarResNet_cfg.arch_parameters.block_parameters.input_injection = False
CifarResNet_cfg.arch_parameters.block_parameters.zero_mean_embedding_kernel = False
CifarResNet_cfg.input_parameters = CN()
CifarResNet_cfg.input_parameters.image_scaler = 1.0
CifarResNet_cfg.input_parameters.reverse_channels = False

default.cfg.model.backbone.CifarResNet = CifarResNet_cfg

def CifarResNet(input_shape=(None, None, None, 3),
                stages=3,
                blocks=2,
                block_repeats=1,
                final_activation=True,
                classification_head=False,
                num_classes=None,
                block_parameters=None,
                use_pretrained=False,
                pretrained_model_file='',
                pretrained_name='CifarResNet',
                name=None):

    """ResNet Version 2 Model builder [b]
        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256
        # Arguments
            input_shape (tensor): shape of input image tensor
            num_stages (int): number of spatial decimation stages
            num_resnet_blocks (int): number of ReSNET blocks at each stage
        # Returns
            model (Model): Keras model instance
        """

    inputs = tf.keras.Input(batch_shape=input_shape)

    # Start model definition.
    in_dim = 16

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    # first perform general feature extraction with convolution
    x = tf.keras.layers.Conv2D(filters=in_dim,
                               kernel_size=3,
                               strides=1,
                               padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Instantiate the stack of residual units
    out_dim = None
    pre_activation = block_parameters['pre_activation']
    for stage in range(stages):
        with tf.name_scope("stage{}".format(stage)):
            for res_block in range(blocks):
                with tf.name_scope("block{}".format(res_block)):
                    # general ReSNET bottleneck block configuration
                    block_parameters['pre_activation'] = pre_activation
                    spatial_decimation = 1
                    out_dim_reduction_rate = 2

                    # first stage has slightly different behavior
                    if stage == 0:
                        out_dim = in_dim * 4
                        out_dim_reduction_rate = 4
                        if res_block == 0:  # first layer and first stage
                            block_parameters['pre_activation'] = False
                    else:
                        out_dim = in_dim * 2

                        if res_block == 0:  # first layer but not first stage
                            spatial_decimation = 2  # downsample


                    # if first layer of a stage
                    if res_block == 0:
                        # linear projection residual shortcut connection
                        x = resMap(out_dim=out_dim,
                                   out_dim_reduction_rate=out_dim_reduction_rate,
                                   spatial_decimation=spatial_decimation,
                                   **block_parameters)(x)
                    else:
                        x = resBlock(out_dim=out_dim,
                                     out_dim_reduction_rate=2,
                                     repeats=block_repeats,
                                     **block_parameters)(x)

        in_dim = out_dim

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    if final_activation:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    outputs = x
    if classification_head:

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(num_classes,
                                        activation='softmax',
                                        kernel_initializer='he_normal')(x)

    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    if use_pretrained:
        base_model = tf.keras.models.load_model(pretrained_model_file, compile=False)
        resnet_backbone = base_model.get_layer(pretrained_name)
        model.set_weights(resnet_backbone.get_weights())

    return model

