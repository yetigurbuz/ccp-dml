import tensorflow as tf

class L1Normalization(tf.keras.layers.Layer):
    """ Keras layer to compute L2 normalization.
    Arguments:
        axis:
    Returns:
      A Keras layer.
    """
    def __init__(self, axis=-1, min_norm=1.0, **kwargs):
        self.axis = axis
        self.min_norm = min_norm
        super(L1Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(L1Normalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        nu = tf.stop_gradient(tf.maximum(self.min_norm,
                                         tf.reduce_sum(inputs, axis=-1, keepdims=True)))
        x = inputs / nu
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(L1Normalization, self).get_config()
        base_config.update({"axis": self.axis,
                            "min_norm": self.min_norm
                            })
        return base_config
