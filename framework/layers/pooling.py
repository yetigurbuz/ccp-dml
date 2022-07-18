import tensorflow as tf

from . import L2Normalization



from ..configs import default
from ..configs.config import CfgNode as CN

# global pooling
GlobalPooling_cfg = CN()
GlobalPooling_cfg.use_average = True
GlobalPooling_cfg.use_max = False
GlobalPooling_cfg.l2_normalize = False

default.cfg.model.embedding_head.GlobalPooling = GlobalPooling_cfg

@tf.keras.utils.register_keras_serializable()
class GlobalPooling(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 use_average=True,
                 use_max=False,
                 name=None,
                 **kwargs):
        super(GlobalPooling, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._use_average = use_average
        self._use_max = use_max

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self._max_pooling = tf.keras.layers.GlobalMaxPooling2D()

        self._transform = tf.keras.layers.Dense(units=embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._maybe_normalize = L2Normalization() if l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)

    def call(self, inputs, **kwargs):

        if self._use_average and self._use_max:
            x_pooled = self._avg_pooling(inputs) + self._max_pooling(inputs)
        else:
            if self._use_average:
                x_pooled = self._avg_pooling(inputs)
            elif self._use_max:
                x_pooled = self._max_pooling(inputs)
            else:
                x_pooled = inputs

        x_emb = self._transform(x_pooled)
        x_emb = self._maybe_normalize(x_emb)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(GlobalPooling, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'use_average': self._use_average,
                       'use_max': self._use_max
                       }
                      )
        return config


