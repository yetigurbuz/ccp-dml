import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class L2Normalization(tf.keras.layers.Layer):
    """ Keras layer to compute L2 normalization.
    Arguments:
        axis:
    Returns:
      A Keras layer.
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(L2Normalization, self).__init__(**kwargs)

    @tf.custom_gradient
    def lipschitzL2Normalize(self, x):
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=self.axis, keepdims=True))
        normalize_mask = tf.cast(tf.greater(x_norm, 1.0), tf.float32)
        identity_mask = 1.0 - normalize_mask
        normalizer = normalize_mask * x_norm + identity_mask
        x_normalized = x / normalizer

        grad_fn = lambda dy: self.gradLipschitzL2Normalize(dy,
                                                           x_normalized=x_normalized,
                                                           normalize_mask=normalize_mask,
                                                           normalizer=normalizer)

        return x_normalized, grad_fn

    @tf.function
    def gradLipschitzL2Normalize(self, dy, x_normalized, normalize_mask, normalizer):

        x_normalized = x_normalized * normalize_mask # make unnormalized vectors 0
        dx = (dy - tf.reduce_sum(x_normalized * dy, axis=-1, keepdims=True) * x_normalized) / normalizer

        return dx

    def build(self, input_shape):
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        #return tf.nn.l2_normalize(inputs, axis=self.axis, epsilon=1e-16)
        return self.lipschitzL2Normalize(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(L2Normalization, self).get_config()
        base_config.update({"axis": self.axis})
        return base_config